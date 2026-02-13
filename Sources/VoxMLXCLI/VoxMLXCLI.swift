import ArgumentParser
import AVFoundation
import Darwin
import Foundation
import VoxMLX

@main
struct VoxMLXCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voxmlx",
        abstract: "Voxtral Mini Realtime speech-to-text in Swift/MLX",
        subcommands: [Transcribe.self, Live.self],
        defaultSubcommand: Transcribe.self
    )
}

extension VoxMLXCLI {
    struct Transcribe: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "transcribe",
            abstract: "Transcribe an audio file."
        )

        @Option(name: .long, help: "Path to audio file (wav/flac/mp3/etc).")
        var audio: String

        @Option(name: .long, help: "Model ID on Hugging Face, or local model directory path.")
        var model: String = VoxtralLoader.defaultModelID

        @Option(name: .long, help: "Sampling temperature (0 = greedy).")
        var temp: Float = 0

        @Option(name: .long, help: "Maximum generated tokens.")
        var maxNewTokens: Int?

        @Flag(name: .long, help: "Print performance stats to stderr.")
        var stats = false

        mutating func run() async throws {
            let audioURL = URL(fileURLWithPath: audio)
            let transcriber = try await loadTranscriber(model: model)
            let result = try transcriber.transcribeWithStats(
                audioURL: audioURL,
                temperature: temp,
                maxNewTokens: maxNewTokens
            )
            print(result.text)

            if stats {
                let s = result.stats
                fputs(
                    String(
                        format: "stats: audio=%.3fs total=%.3fs mel=%.3fs encode=%.3fs prefill=%.3fs decode=%.3fs tokens=%d tok/s=%.2f\n",
                        s.audioSeconds,
                        s.totalSeconds,
                        s.melSeconds,
                        s.encodeSeconds,
                        s.prefillSeconds,
                        s.decodeSeconds,
                        s.generatedTokenCount,
                        s.generatedTokensPerSecond
                    ),
                    stderr
                )
            }
        }
    }

    struct Live: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "live",
            abstract: "Realtime transcription from microphone."
        )

        @Option(name: .long, help: "Model ID on Hugging Face, or local model directory path.")
        var model: String = VoxtralLoader.defaultRealtimeModelID

        @Option(name: .long, help: "Sampling temperature (0 = greedy).")
        var temp: Float = 0

        @Option(name: .long, help: "Streaming chunk duration in milliseconds (rounded to 80 ms multiples).")
        var chunkMs: Float = 80

        @Option(name: .long, help: "Override transcription delay in milliseconds.")
        var transcriptionDelayMs: Float?

        @Option(name: .long, help: "Right-pad token count used when finalizing stream.")
        var rightPadTokens: Int = 17

        @Option(name: .long, help: "Decoder KV sliding window size (tokens). Smaller = lower latency and memory.")
        var decoderWindow: Int = 2048

        @Option(name: .long, help: "Max buffered microphone backlog before dropping stale audio (ms).")
        var maxBacklogMs: Float = 400

        mutating func run() async throws {
            let transcriber = try await loadTranscriber(model: model)
            let session = try VoxtralRealtimeSession(
                transcriber: transcriber,
                temperature: temp,
                chunkDurationMs: chunkMs,
                transcriptionDelayMs: transcriptionDelayMs,
                rightPadTokens: rightPadTokens,
                decoderWindowTokens: decoderWindow
            )
            try runMicrophone(session: session, maxBacklogMs: maxBacklogMs)
        }

        private func runMicrophone(session: VoxtralRealtimeSession, maxBacklogMs: Float) throws {
            let source = try MicrophoneChunkSource(chunkSamples: session.chunkSamples)
            print("Listening... (Ctrl+C to stop)")

            try source.start()
            defer {
                source.stop()
            }

            let maxBacklogChunks = max(1, Int(Foundation.ceil(maxBacklogMs / session.tokenDurationMs)))
            while true {
                if let chunk = source.popChunk(maxBacklogChunks: maxBacklogChunks) {
                    emit(session.appendAudioSamples(chunk))
                } else {
                    Thread.sleep(forTimeInterval: 0.002)
                }
            }
        }

        private func emit(_ text: String) {
            guard !text.isEmpty else { return }
            FileHandle.standardOutput.write(Data(text.utf8))
            fflush(stdout)
        }
    }
}

private func loadTranscriber(model: String) async throws -> VoxtralTranscriber {
    if FileManager.default.fileExists(atPath: model) {
        return try VoxtralTranscriber.load(from: URL(fileURLWithPath: model))
    } else {
        return try await VoxtralTranscriber.load(modelID: model)
    }
}

private final class MicrophoneChunkSource {
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private let chunkSamples: Int
    private var chunkQueue: [[Float]] = []
    private var chunkQueueStart = 0
    private var sampleBuffer: [Float] = []
    private var sampleBufferStart = 0

    init(chunkSamples: Int) throws {
        self.chunkSamples = chunkSamples
    }

    func start() throws {
        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let tapFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputFormat.sampleRate,
            channels: inputFormat.channelCount,
            interleaved: false
        ) else {
            throw MicrophoneChunkSourceError.invalidTapFormat
        }

        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(chunkSamples), format: tapFormat) { [weak self] buffer, _ in
            self?.handleBuffer(buffer)
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    func popChunk(maxBacklogChunks: Int) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        guard chunkQueueStart < chunkQueue.count else {
            chunkQueue.removeAll(keepingCapacity: true)
            chunkQueueStart = 0
            return nil
        }
        let backlog = chunkQueue.count - chunkQueueStart
        if backlog > maxBacklogChunks {
            let toDrop = backlog - maxBacklogChunks
            chunkQueueStart += toDrop
        }
        let chunk = chunkQueue[chunkQueueStart]
        chunkQueueStart += 1
        if chunkQueueStart > 64, chunkQueueStart * 2 >= chunkQueue.count {
            chunkQueue.removeFirst(chunkQueueStart)
            chunkQueueStart = 0
        }
        return chunk
    }

    func consumeRemainder() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let remainder: [Float]
        if sampleBufferStart < sampleBuffer.count {
            remainder = Array(sampleBuffer[sampleBufferStart...])
        } else {
            remainder = []
        }
        sampleBuffer.removeAll(keepingCapacity: true)
        sampleBufferStart = 0
        return remainder
    }

    private func handleBuffer(_ buffer: AVAudioPCMBuffer) {
        let frames = Int(buffer.frameLength)
        guard frames > 0, let channelData = buffer.floatChannelData else { return }

        let channels = Int(buffer.format.channelCount)
        var mono = Array(repeating: Float(0), count: frames)

        if channels == 1 {
            let src = channelData[0]
            for i in 0 ..< frames {
                mono[i] = src[i]
            }
        } else {
            for c in 0 ..< channels {
                let src = channelData[c]
                for i in 0 ..< frames {
                    mono[i] += src[i]
                }
            }
            let scale = 1.0 / Float(channels)
            for i in 0 ..< frames {
                mono[i] *= scale
            }
        }

        let inputRate = Int(round(buffer.format.sampleRate))
        let resampled = VoxtralAudio.resampleToModelRate(mono, inputRate: inputRate)

        lock.lock()
        sampleBuffer.append(contentsOf: resampled)
        while sampleBuffer.count - sampleBufferStart >= chunkSamples {
            let start = sampleBufferStart
            let end = start + chunkSamples
            chunkQueue.append(Array(sampleBuffer[start ..< end]))
            sampleBufferStart = end
        }
        if sampleBufferStart > 32_768, sampleBufferStart * 2 >= sampleBuffer.count {
            sampleBuffer.removeFirst(sampleBufferStart)
            sampleBufferStart = 0
        }
        lock.unlock()
    }
}

private enum MicrophoneChunkSourceError: Error {
    case invalidTapFormat
}
