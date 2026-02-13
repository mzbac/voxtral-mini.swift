import Foundation
import Hub
import MLX

public enum VoxtralTranscriberError: Error {
    case audioTooShortForPrompt(required: Int, available: Int)
    case missingStreamingPadToken
}

public struct VoxtralTranscriptionStats: Sendable {
    public let audioSeconds: Double
    public let loadAudioSeconds: Double
    public let melSeconds: Double
    public let encodeSeconds: Double
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let totalSeconds: Double
    public let generatedTokenCount: Int
    public let generatedTokensPerSecond: Double
}

public struct VoxtralTranscriptionResult: Sendable {
    public let text: String
    public let stats: VoxtralTranscriptionStats
}

public final class VoxtralTranscriber: @unchecked Sendable {
    public let model: VoxtralRealtime
    public let tokenizer: TekkenTokenizer
    public let config: VoxtralParams

    public init(model: VoxtralRealtime, tokenizer: TekkenTokenizer, config: VoxtralParams) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    public static func load(
        modelID: String = VoxtralLoader.defaultModelID,
        hubApi: HubApi? = nil
    ) async throws -> VoxtralTranscriber {
        let loaded = try await VoxtralLoader.load(modelID: modelID, hubApi: hubApi)
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    }

    public static func load(from directory: URL) throws -> VoxtralTranscriber {
        let loaded = try VoxtralLoader.load(directory: directory)
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    }

    public func transcribe(
        audioURL: URL,
        temperature: Float = 0,
        maxNewTokens: Int? = nil
    ) throws -> String {
        try transcribeWithStats(
            audioURL: audioURL,
            temperature: temperature,
            maxNewTokens: maxNewTokens
        ).text
    }

    public func transcribeWithStats(
        audioURL: URL,
        temperature: Float = 0,
        maxNewTokens: Int? = nil
    ) throws -> VoxtralTranscriptionResult {
        let runStart = DispatchTime.now().uptimeNanoseconds

        let loadStart = DispatchTime.now().uptimeNanoseconds
        let audio = try VoxtralAudio.loadAudio(url: audioURL)
        let loadEnd = DispatchTime.now().uptimeNanoseconds

        let prompt = try buildPromptTokens()
        let promptTokens = prompt.tokens
        let prefixLength = promptTokens.count
        let nDelayTokens = prompt.nDelayTokens

        let melStart = DispatchTime.now().uptimeNanoseconds
        let paddedAudio = VoxtralAudio.padAudio(
            audio,
            leftPadTokens: prompt.leftPadTokens,
            rightPadTokens: prompt.rightPadTokens
        )
        let mel = VoxtralAudio.logMelSpectrogram(paddedAudio)
        let melEnd = DispatchTime.now().uptimeNanoseconds

        let encodeStart = DispatchTime.now().uptimeNanoseconds
        let audioEmbeds = model.encode(mel)
        eval(audioEmbeds)
        let encodeEnd = DispatchTime.now().uptimeNanoseconds

        let availableAudioTokens = audioEmbeds.dim(0)
        if availableAudioTokens < prefixLength {
            throw VoxtralTranscriberError.audioTooShortForPrompt(
                required: prefixLength,
                available: availableAudioTokens
            )
        }

        let tCond = model.timeEmbedding(MLXArray([Float(nDelayTokens)]))
        eval(tCond)

        let promptIDs = MLXArray(promptTokens.map(Int32.init)).reshaped(1, prefixLength)
        let textEmbeds = model.languageModel.embed(promptIDs)
        let adaScales = model.languageModel.makeAdaScales(tCond: tCond, dtype: textEmbeds.dtype)
        eval(adaScales)

        let prefixAudioEmbeds = audioEmbeds[0 ..< prefixLength, 0...].reshaped(1, prefixLength, config.dim)
        let prefixEmbeds = textEmbeds + prefixAudioEmbeds

        let cache = model.makeDecoderCache(maxSize: config.slidingWindow)

        let prefillStart = DispatchTime.now().uptimeNanoseconds
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
        let prefillEnd = DispatchTime.now().uptimeNanoseconds

        let lastLogits = model.languageModel.project(lastHidden)
        var token = sampleToken(from: lastLogits, temperature: temperature)
        asyncEval(token)
        var outputTokens: [Int] = []

        let totalDecodablePositions = availableAudioTokens - prefixLength
        let maxOutputTokens = min(
            maxNewTokens ?? (totalDecodablePositions + 1),
            totalDecodablePositions + 1
        )

        let decodeStart = DispatchTime.now().uptimeNanoseconds
        if maxOutputTokens > 0 {
            var emitted = 0
            while emitted < maxOutputTokens {
                let tokenID = token.item(Int.self)
                if tokenID == tokenizer.eosID {
                    break
                }
                outputTokens.append(tokenID)
                emitted += 1

                if emitted >= maxOutputTokens {
                    break
                }

                let position = prefixLength + emitted - 1
                if position >= availableAudioTokens {
                    break
                }

                let nextToken = decodeStep(
                    token: token,
                    position: position,
                    audioEmbeds: audioEmbeds,
                    tCond: tCond,
                    cache: cache,
                    temperature: temperature,
                    adaScales: adaScales
                )
                asyncEval(nextToken)
                token = nextToken
            }
        }
        let decodeEnd = DispatchTime.now().uptimeNanoseconds

        if let last = outputTokens.last, last == tokenizer.eosID {
            outputTokens.removeLast()
        }

        let text = tokenizer.decode(outputTokens, ignoreSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let runEnd = DispatchTime.now().uptimeNanoseconds

        func seconds(_ start: UInt64, _ end: UInt64) -> Double {
            Double(end - start) / 1_000_000_000.0
        }

        let decodeSeconds = seconds(decodeStart, decodeEnd)
        let generatedTokenCount = outputTokens.count
        let stats = VoxtralTranscriptionStats(
            audioSeconds: Double(audio.count) / Double(VoxtralAudio.sampleRate),
            loadAudioSeconds: seconds(loadStart, loadEnd),
            melSeconds: seconds(melStart, melEnd),
            encodeSeconds: seconds(encodeStart, encodeEnd),
            prefillSeconds: seconds(prefillStart, prefillEnd),
            decodeSeconds: decodeSeconds,
            totalSeconds: seconds(runStart, runEnd),
            generatedTokenCount: generatedTokenCount,
            generatedTokensPerSecond: decodeSeconds > 0 ? Double(generatedTokenCount) / decodeSeconds : 0
        )

        return VoxtralTranscriptionResult(text: text, stats: stats)
    }

    private func buildPromptTokens() throws -> (
        tokens: [Int], nDelayTokens: Int, leftPadTokens: Int, rightPadTokens: Int
    ) {
        guard let streamingPad = tokenizer.specialTokenID("[STREAMING_PAD]") else {
            throw VoxtralTranscriberError.missingStreamingPadToken
        }

        let leftPadTokens = tokenizer.audioMetadata.streamingNLeftPadTokens ?? 32

        // Each audio token spans 1280 samples at 16 kHz -> 80 ms.
        let tokenDurationMs = (Float(VoxtralAudio.samplesPerToken) * 1000.0) / Float(VoxtralAudio.sampleRate)
        let transcriptionDelayMs = tokenizer.audioMetadata.transcriptionDelayMs ?? 480.0
        let nDelayTokens = max(0, Int(round(transcriptionDelayMs / tokenDurationMs)))

        let prefixCount = leftPadTokens + nDelayTokens
        let tokens = [tokenizer.bosID] + Array(repeating: streamingPad, count: prefixCount)
        let offlineStreamingBufferTokens = 10
        let rightPadTokens = (nDelayTokens + 1) + offlineStreamingBufferTokens
        return (tokens, nDelayTokens, leftPadTokens, rightPadTokens)
    }

    private func decodeStep(
        token: MLXArray,
        position: Int,
        audioEmbeds: MLXArray,
        tCond: MLXArray,
        cache: [RotatingKVCache],
        temperature: Float,
        adaScales: [MLXArray]?
    ) -> MLXArray {
        let tokenIDs = token.reshaped(1, 1)
        let tokenEmbed = model.languageModel.embed(tokenIDs)

        let audioPos = audioEmbeds[position ..< (position + 1), 0...].reshaped(1, 1, config.dim)
        let stepEmbed = audioPos + tokenEmbed

        let hidden = model.languageModel.forwardHidden(
            stepEmbed,
            tCond: tCond,
            mask: .none,
            cache: cache,
            adaScales: adaScales
        )

        let logits = model.languageModel.project(hidden)
        return sampleToken(from: logits, temperature: temperature)
    }

    private func sampleToken(from logits: MLXArray, temperature: Float) -> MLXArray {
        let lastPosition = logits.dim(1) - 1
        let lastLogits = logits[0..., lastPosition ..< (lastPosition + 1), 0...]

        if temperature <= 0 {
            return argMax(lastLogits, axis: -1).asType(.int32)
        }

        return categorical(lastLogits * (1.0 / temperature)).asType(.int32)
    }
}
