import Foundation
import MLX

public enum VoxtralRealtimeSessionError: Error {
    case invalidChunkDurationMs(Float)
}

public final class VoxtralRealtimeSession: @unchecked Sendable {
    public let transcriber: VoxtralTranscriber
    public let temperature: Float
    public let chunkSamples: Int
    public let tokenDurationMs: Float
    public let rightPadTokens: Int
    public let decoderWindowTokens: Int

    private let model: VoxtralRealtime
    private let tokenizer: TekkenTokenizer
    private let config: VoxtralParams

    private let prefixLength: Int
    private let leftPadTokens: Int
    private let eosID: Int
    private let tCond: MLXArray
    private let textEmbeds: MLXArray
    private let adaScales: [MLXArray]

    private var pendingAudio: [Float] = []
    private var pendingAudioStart = 0
    private var audioTail: [Float]?
    private var conv1Tail: MLXArray?
    private var conv2Tail: MLXArray?
    private var encoderCache: [RotatingKVCache]?
    private var downsampleBuffer: MLXArray?

    private var audioEmbeds: MLXArray?
    private var decoderCache: [RotatingKVCache]?
    private var currentToken: MLXArray?

    private var totalAudioSamplesFed: Int = 0
    private var totalDecodedPositions: Int = 0
    private var firstCycle = true
    private var prefilled = false

    private var pendingDecodedBytes = Data()

    public init(
        transcriber: VoxtralTranscriber,
        temperature: Float = 0,
        chunkDurationMs: Float = 80,
        transcriptionDelayMs: Float? = nil,
        rightPadTokens: Int = 17,
        decoderWindowTokens: Int? = nil
    ) throws {
        self.transcriber = transcriber
        model = transcriber.model
        tokenizer = transcriber.tokenizer
        config = transcriber.config
        self.temperature = temperature
        self.rightPadTokens = rightPadTokens
        self.decoderWindowTokens = max(256, decoderWindowTokens ?? config.slidingWindow)

        tokenDurationMs = (Float(VoxtralAudio.samplesPerToken) * 1000.0) / Float(VoxtralAudio.sampleRate)
        if chunkDurationMs <= 0 {
            throw VoxtralRealtimeSessionError.invalidChunkDurationMs(chunkDurationMs)
        }
        let chunkTokenCount = max(1, Int(round(chunkDurationMs / tokenDurationMs)))
        chunkSamples = chunkTokenCount * VoxtralAudio.samplesPerToken

        guard let streamingPad = tokenizer.specialTokenID("[STREAMING_PAD]") else {
            throw VoxtralTranscriberError.missingStreamingPadToken
        }

        leftPadTokens = tokenizer.audioMetadata.streamingNLeftPadTokens ?? 32
        let delayMs = transcriptionDelayMs ?? tokenizer.audioMetadata.transcriptionDelayMs ?? 480.0
        let nDelayTokens = max(0, Int(round(delayMs / tokenDurationMs)))

        let promptTokens = [tokenizer.bosID] + Array(repeating: streamingPad, count: leftPadTokens + nDelayTokens)
        prefixLength = promptTokens.count
        eosID = tokenizer.eosID

        let promptIDs = MLXArray(promptTokens.map(Int32.init)).reshaped(1, prefixLength)
        textEmbeds = model.languageModel.embed(promptIDs)
        tCond = model.timeEmbedding(MLXArray([Float(nDelayTokens)]))
        adaScales = model.languageModel.makeAdaScales(tCond: tCond, dtype: textEmbeds.dtype)
        eval([textEmbeds, tCond] + adaScales)
    }

    public func appendAudioSamples(_ samples: [Float]) -> String {
        if !samples.isEmpty {
            pendingAudio.append(contentsOf: samples)
            consumePendingAudio()
        }
        return decodeAvailable(decodeAllAvailable: false)
    }

    public func finishStream() -> String {
        pendingAudio.append(contentsOf: Array(repeating: Float(0), count: rightPadTokens * VoxtralAudio.samplesPerToken))
        consumePendingAudio()

        var out = decodeAvailable(decodeAllAvailable: true)

        if let token = currentToken {
            let tokenID = token.item(Int.self)
            if tokenID != eosID {
                out += emitTokenFragment(tokenID)
            }
        }

        out += flushPendingDecodedBytes()
        resetAllState()
        return out
    }

    private func consumePendingAudio() {
        let available = pendingAudio.count - pendingAudioStart
        if firstCycle {
            guard available >= chunkSamples else { return }

            let nFeed = (available / chunkSamples) * chunkSamples
            guard nFeed > 0 else { return }

            let leftPad = Array(repeating: Float(0), count: leftPadTokens * VoxtralAudio.samplesPerToken)
            let chunk = leftPad + consumePendingAudioPrefix(nFeed)

            totalAudioSamplesFed += nFeed
            appendEmbeddings(from: chunk)
            firstCycle = false
            return
        }

        let nFeed = (available / chunkSamples) * chunkSamples
        guard nFeed > 0 else { return }

        let chunk = consumePendingAudioPrefix(nFeed)

        totalAudioSamplesFed += nFeed
        appendEmbeddings(from: chunk)
    }

    private func appendEmbeddings(from audioChunk: [Float]) {
        let melStep = VoxtralAudio.logMelSpectrogramStep(audioChunk: audioChunk, audioTail: audioTail)
        audioTail = melStep.newTail

        let encodedStep = model.encodeStep(
            newMel: melStep.mel,
            conv1Tail: conv1Tail,
            conv2Tail: conv2Tail,
            encoderCache: encoderCache,
            downsampleBuffer: downsampleBuffer
        )

        conv1Tail = encodedStep.conv1Tail
        conv2Tail = encodedStep.conv2Tail
        encoderCache = encodedStep.encoderCache
        downsampleBuffer = encodedStep.downsampleBuffer

        guard let newAudioEmbeds = encodedStep.newAudioEmbeds else { return }

        if let audioEmbeds {
            let merged = concatenated([audioEmbeds, newAudioEmbeds], axis: 0)
            eval(merged)
            self.audioEmbeds = merged
        } else {
            eval(newAudioEmbeds)
            self.audioEmbeds = newAudioEmbeds
        }
    }

    private func decodeAvailable(decodeAllAvailable: Bool) -> String {
        guard var embeds = audioEmbeds else { return "" }

        if !prefilled {
            guard embeds.dim(0) >= prefixLength else { return "" }

            let prefixAudioEmbeds = embeds[0 ..< prefixLength, 0...].reshaped(1, prefixLength, config.dim)
            let prefixEmbeds = textEmbeds + prefixAudioEmbeds

            let cache = model.makeDecoderCache(maxSize: decoderWindowTokens)
            let prefillHidden = model.languageModel.forwardHidden(
                prefixEmbeds,
                tCond: tCond,
                mask: .causal,
                cache: cache,
                adaScales: adaScales
            )
            let lastPosition = prefillHidden.dim(1) - 1
            let lastHidden = prefillHidden[0..., lastPosition ..< (lastPosition + 1), 0...]
            let cacheArrays = cache.flatMap { layer in
                [layer.keys, layer.values]
            }.compactMap { $0 }
            eval([lastHidden] + cacheArrays)

            let lastLogits = model.languageModel.project(lastHidden)
            let token = sampleToken(from: lastLogits)
            asyncEval(token)

            decoderCache = cache
            currentToken = token
            totalDecodedPositions = prefixLength
            prefilled = true

            if embeds.dim(0) > prefixLength {
                embeds = embeds[prefixLength..., 0...]
                audioEmbeds = embeds
            } else {
                audioEmbeds = nil
                return ""
            }
        }

        guard let cache = decoderCache, var token = currentToken, let pendingEmbeds = audioEmbeds else {
            return ""
        }

        let nDecodable: Int
        if decodeAllAvailable {
            nDecodable = pendingEmbeds.dim(0)
        } else {
            let safeTotal = leftPadTokens + (totalAudioSamplesFed / VoxtralAudio.samplesPerToken)
            nDecodable = min(pendingEmbeds.dim(0), safeTotal - totalDecodedPositions)
        }

        guard nDecodable > 0 else { return "" }

        var textOut = ""
        var consumed = 0

        for idx in 0 ..< nDecodable {
            let tokenIDs = token.reshaped(1, 1)
            let tokenEmbed = model.languageModel.embed(tokenIDs)
            let audioPos = pendingEmbeds[idx ..< (idx + 1), 0...].reshaped(1, 1, config.dim)
            let stepEmbed = audioPos + tokenEmbed

            let hidden = model.languageModel.forwardHidden(
                stepEmbed,
                tCond: tCond,
                mask: .none,
                cache: cache,
                adaScales: adaScales
            )
            let logits = model.languageModel.project(hidden)
            let nextToken = sampleToken(from: logits)
            asyncEval(nextToken)

            let tokenID = token.item(Int.self)
            consumed += 1

            if tokenID == eosID {
                textOut += "\n"
                resetAllState()
                break
            }

            textOut += emitTokenFragment(tokenID)

            token = nextToken
            currentToken = nextToken
        }

        if prefilled {
            if let remaining = audioEmbeds {
                if consumed >= remaining.dim(0) {
                    audioEmbeds = nil
                } else {
                    audioEmbeds = remaining[consumed..., 0...]
                }
            }
            totalDecodedPositions += consumed
        }

        return textOut
    }

    private func consumePendingAudioPrefix(_ count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let start = pendingAudioStart
        let end = start + count
        let chunk = Array(pendingAudio[start ..< end])
        pendingAudioStart = end

        if pendingAudioStart == pendingAudio.count {
            pendingAudio.removeAll(keepingCapacity: true)
            pendingAudioStart = 0
        } else if pendingAudioStart > 32_768, pendingAudioStart * 2 >= pendingAudio.count {
            pendingAudio.removeFirst(pendingAudioStart)
            pendingAudioStart = 0
        }

        return chunk
    }

    private func emitTokenFragment(_ tokenID: Int) -> String {
        pendingDecodedBytes.append(
            tokenizer.decodedBytes(for: tokenID, ignoreSpecialTokens: true)
        )
        guard !pendingDecodedBytes.isEmpty else { return "" }

        if let text = String(data: pendingDecodedBytes, encoding: .utf8) {
            pendingDecodedBytes.removeAll(keepingCapacity: true)
            return text
        }
        return ""
    }

    private func flushPendingDecodedBytes() -> String {
        guard !pendingDecodedBytes.isEmpty else { return "" }
        let text = String(decoding: pendingDecodedBytes, as: UTF8.self)
        pendingDecodedBytes.removeAll(keepingCapacity: true)
        return text
    }

    private func sampleToken(from logits: MLXArray) -> MLXArray {
        let lastPosition = logits.dim(1) - 1
        let lastLogits = logits[0..., lastPosition ..< (lastPosition + 1), 0...]
        if temperature <= 0 {
            return argMax(lastLogits, axis: -1).asType(.int32)
        }
        return categorical(lastLogits * (1.0 / temperature)).asType(.int32)
    }

    private func resetAllState() {
        pendingAudio.removeAll(keepingCapacity: true)
        pendingAudioStart = 0
        audioTail = nil
        conv1Tail = nil
        conv2Tail = nil
        encoderCache = nil
        downsampleBuffer = nil

        audioEmbeds = nil
        decoderCache = nil
        currentToken = nil

        totalAudioSamplesFed = 0
        totalDecodedPositions = 0
        firstCycle = true
        prefilled = false

        pendingDecodedBytes.removeAll(keepingCapacity: true)
    }
}
