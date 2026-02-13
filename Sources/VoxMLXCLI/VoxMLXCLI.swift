import ArgumentParser
import AVFoundation
import Darwin
import Foundation
import MLX
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
    struct SharedOptions: ParsableArguments {
        @Option(name: .long, help: "MLX allocator cache limit in MB (0 disables cache).")
        var mlxCacheLimitMb: Int = 2048

        @Flag(name: .shortAndLong, help: "Enable verbose setup logs (permissions/model/cache/download).")
        var verbose = false
    }

    struct Transcribe: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "transcribe",
            abstract: "Transcribe an audio file."
        )

        @OptionGroup
        var shared: SharedOptions

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
            let cacheBytes = try configureMLXCacheLimit(cacheLimitMB: shared.mlxCacheLimitMb)
            logVerbose(shared.verbose, "mlx cache limit: \(shared.mlxCacheLimitMb) MB (\(cacheBytes) bytes)")
            let audioURL = URL(fileURLWithPath: audio)
            let transcriber = try await loadTranscriber(model: model, verbose: shared.verbose)
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

        @OptionGroup
        var shared: SharedOptions

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
            try await ensureMicrophonePermission(verbose: shared.verbose)
            let cacheBytes = try configureMLXCacheLimit(cacheLimitMB: shared.mlxCacheLimitMb)
            logVerbose(shared.verbose, "mlx cache limit: \(shared.mlxCacheLimitMb) MB (\(cacheBytes) bytes)")
            let transcriber = try await loadTranscriber(model: model, verbose: shared.verbose)
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

private func ensureMicrophonePermission(verbose: Bool) async throws {
    let deniedMessage =
        "Microphone access is required for `live`. Enable it in System Settings > Privacy & Security > Microphone."

    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        logVerbose(verbose, "microphone permission: authorized")
        return
    case .notDetermined:
        logVerbose(verbose, "microphone permission: requesting access...")
        let granted = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
        if !granted {
            throw CleanExit.message(deniedMessage)
        }
        logVerbose(verbose, "microphone permission: granted")
    case .denied, .restricted:
        logVerbose(verbose, "microphone permission: denied/restricted")
        throw CleanExit.message(deniedMessage)
    @unknown default:
        throw CleanExit.message("Unable to determine microphone permission status.")
    }
}

private func configureMLXCacheLimit(cacheLimitMB: Int) throws -> Int {
    guard cacheLimitMB >= 0 else {
        throw ValidationError("--mlx-cache-limit-mb must be >= 0.")
    }
    let (bytes, overflow) = cacheLimitMB.multipliedReportingOverflow(by: 1024 * 1024)
    guard !overflow else {
        throw ValidationError("--mlx-cache-limit-mb is too large.")
    }
    Memory.cacheLimit = bytes
    return bytes
}

private func loadTranscriber(model: String, verbose: Bool) async throws -> VoxtralTranscriber {
    if FileManager.default.fileExists(atPath: model) {
        let modelDirectory = URL(fileURLWithPath: model)
        logVerbose(verbose, "model source: local directory \(modelDirectory.path)")
        let loaded = try VoxtralLoader.load(directory: modelDirectory)
        logVerbose(verbose, "model loaded from: \(loaded.directory.path)")
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    } else {
        logVerbose(verbose, "model source: Hugging Face id \(model)")
        logVerbose(verbose, "downloading model files if needed...")

        final class ProgressState {
            var lastBucket: Int = -1
        }
        let state = ProgressState()
        let loaded = try await VoxtralLoader.load(modelID: model) { progress, speed in
            guard verbose, progress.totalUnitCount > 0 else { return }
            let fraction = max(0.0, min(1.0, progress.fractionCompleted))
            let percent = Int((fraction * 100.0).rounded())
            let bucket = percent / 5
            guard bucket > state.lastBucket || percent >= 100 else { return }
            state.lastBucket = bucket

            if let speed, speed > 0 {
                fputs(
                    String(
                        format: "[voxmlx] model download: %3d%% (%.1f MiB/s)\n",
                        percent,
                        speed / (1024.0 * 1024.0)
                    ),
                    stderr
                )
            } else {
                fputs(String(format: "[voxmlx] model download: %3d%%\n", percent), stderr)
            }
        }
        logVerbose(verbose, "model loaded from cache directory: \(loaded.directory.path)")
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    }
}

private func logVerbose(_ enabled: Bool, _ message: @autoclosure () -> String) {
    guard enabled else { return }
    fputs("[voxmlx] \(message())\n", stderr)
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
