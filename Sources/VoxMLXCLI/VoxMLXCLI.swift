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

        @Option(
            name: .long,
            help: "Hugging Face cache root. Precedence: --download-base-path > HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub."
        )
        var downloadBasePath: String?

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

        @Flag(
            name: .long,
            help: "Feed the file through the realtime streaming session (same path as `live`) in chunks and print text incrementally."
        )
        var realtime = false

        @Option(name: .long, help: "Streaming chunk duration in milliseconds (with --realtime).")
        var chunkMs: Float = 80

        mutating func run() async throws {
            let cacheBytes = try configureMLXCacheLimit(cacheLimitMB: shared.mlxCacheLimitMb)
            logVerbose(shared.verbose, "mlx cache limit: \(shared.mlxCacheLimitMb) MB (\(cacheBytes) bytes)")
            let audioPath = NSString(string: audio).expandingTildeInPath
            guard FileManager.default.fileExists(atPath: audioPath) else {
                throw ValidationError("Audio file not found: \(audio)")
            }
            let audioURL = URL(fileURLWithPath: audioPath)
            let transcriber = try await loadTranscriber(
                model: model,
                verbose: shared.verbose,
                downloadBasePath: shared.downloadBasePath
            )

            if realtime {
                try runRealtimeSimulation(transcriber: transcriber, audioURL: audioURL)
                return
            }

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

        private func runRealtimeSimulation(transcriber: VoxtralTranscriber, audioURL: URL) throws {
            let session = try VoxtralRealtimeSession(
                transcriber: transcriber,
                temperature: temp,
                chunkDurationMs: chunkMs
            )
            let samples = try VoxtralAudio.loadAudio(url: audioURL)

            var cursor = 0
            var chunkIndex = 0
            var maxChunkSeconds = 0.0
            var chunksOverBudget = 0
            var reportedChunks: [(index: Int, seconds: Double)] = []
            let start = DispatchTime.now().uptimeNanoseconds
            while cursor < samples.count {
                let end = min(samples.count, cursor + session.chunkSamples)
                let chunkStart = DispatchTime.now().uptimeNanoseconds
                let text = session.appendAudioSamples(Array(samples[cursor ..< end]))
                let chunkSeconds = Double(DispatchTime.now().uptimeNanoseconds - chunkStart) / 1e9
                maxChunkSeconds = max(maxChunkSeconds, chunkSeconds)
                let overBudget = chunkSeconds > Double(session.tokenDurationMs) / 1000.0
                if overBudget {
                    chunksOverBudget += 1
                }
                // Verbose mode lists the warm-up region (first chunks) plus any stalls.
                if overBudget || chunkIndex < 12 {
                    reportedChunks.append((chunkIndex, chunkSeconds))
                }
                emit(text)
                cursor = end
                chunkIndex += 1
            }
            emit(session.finishStream())
            emit("\n")
            let total = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

            if stats {
                let audioSeconds = Double(samples.count) / Double(VoxtralAudio.sampleRate)
                fputs(
                    String(
                        format: "stats: audio=%.3fs total=%.3fs realtime_factor=%.2fx chunks=%d max_chunk=%.3fs chunks_over_budget=%d\n",
                        audioSeconds,
                        total,
                        audioSeconds / max(total, 1e-9),
                        chunkIndex,
                        maxChunkSeconds,
                        chunksOverBudget
                    ),
                    stderr
                )
            }
            if shared.verbose {
                for entry in reportedChunks.prefix(32) {
                    fputs(String(format: "[voxmlx] chunk %d took %.3fs\n", entry.index, entry.seconds), stderr)
                }
            }
        }

        private func emit(_ text: String) {
            guard !text.isEmpty else { return }
            FileHandle.standardOutput.write(Data(text.utf8))
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

        @Option(
            name: .long,
            help: "Max buffered microphone backlog before dropping stale audio (ms). The session runs several times faster than realtime, so a generous limit lets it catch up after a stall instead of cutting a hole in the speech."
        )
        var maxBacklogMs: Float = 3000

        mutating func run() async throws {
            guard chunkMs > 0 else {
                throw ValidationError("--chunk-ms must be > 0.")
            }
            guard rightPadTokens >= 0 else {
                throw ValidationError("--right-pad-tokens must be >= 0.")
            }
            guard decoderWindow > 0 else {
                throw ValidationError("--decoder-window must be > 0.")
            }
            guard maxBacklogMs > 0 else {
                throw ValidationError("--max-backlog-ms must be > 0.")
            }

            try await ensureMicrophonePermission(verbose: shared.verbose)
            let cacheBytes = try configureMLXCacheLimit(cacheLimitMB: shared.mlxCacheLimitMb)
            logVerbose(shared.verbose, "mlx cache limit: \(shared.mlxCacheLimitMb) MB (\(cacheBytes) bytes)")
            let transcriber = try await loadTranscriber(
                model: model,
                verbose: shared.verbose,
                downloadBasePath: shared.downloadBasePath
            )
            let session = try VoxtralRealtimeSession(
                transcriber: transcriber,
                temperature: temp,
                chunkDurationMs: chunkMs,
                transcriptionDelayMs: transcriptionDelayMs,
                rightPadTokens: rightPadTokens,
                decoderWindowTokens: decoderWindow
            )
            try runMicrophone(session: session, maxBacklogMs: maxBacklogMs, verbose: shared.verbose)
        }

        private func runMicrophone(session: VoxtralRealtimeSession, maxBacklogMs: Float, verbose: Bool) throws {
            let source = try MicrophoneChunkSource(chunkSamples: session.chunkSamples)
            let shutdown = ShutdownSignal()
            shutdown.install { source.requestStop() }

            // Pay first-use costs (Metal pipelines, weight page-in) before any speech is
            // captured; otherwise the first chunks stall, audio queues up or gets dropped,
            // and the model decodes a garbled start.
            fputs("Warming up model...\n", stderr)
            let warmSeconds = session.warmUp()
            logVerbose(verbose, String(format: "warm-up took %.2fs", warmSeconds))

            try source.start()
            logVerbose(verbose, "microphone input: \(source.inputDescription)")
            fputs("Listening... (Ctrl+C to stop)\n", stderr)

            let maxBacklogChunks = max(1, Int(Foundation.ceil(maxBacklogMs / session.tokenDurationMs)))
            var reportedDrops = 0
            while !source.stopRequested {
                guard let chunk = source.popChunk(maxBacklogChunks: maxBacklogChunks, timeout: 0.25) else {
                    continue
                }
                emit(session.appendAudioSamples(chunk))

                let dropped = source.droppedChunkCount
                if dropped > reportedDrops {
                    let ms = Float(dropped - reportedDrops) * session.tokenDurationMs
                    fputs(
                        String(format: "\n[voxmlx] warning: dropped %.0f ms of audio, transcription cannot keep up\n", ms),
                        stderr
                    )
                    reportedDrops = dropped
                }
            }

            // Ctrl+C: stop capturing, then flush whatever audio and tokens are still pending
            // so the tail of the utterance is not lost.
            source.stop()
            let dropped = source.droppedChunkCount
            for chunk in source.drainChunks() {
                emit(session.appendAudioSamples(chunk))
            }
            let remainder = source.consumeRemainder()
            if !remainder.isEmpty {
                emit(session.appendAudioSamples(remainder))
            }
            emit(session.finishStream())
            emit("\n")
            if dropped > 0 {
                logVerbose(verbose, "dropped \(dropped) stale microphone chunk(s) in total")
            }
        }

        private func emit(_ text: String) {
            guard !text.isEmpty else { return }
            FileHandle.standardOutput.write(Data(text.utf8))
        }
    }
}

/// Turns SIGINT/SIGTERM into a cooperative stop. A second signal exits immediately.
private final class ShutdownSignal: @unchecked Sendable {
    private var sources: [DispatchSourceSignal] = []
    private let lock = NSLock()
    private var fired = false

    func install(_ onStop: @escaping @Sendable () -> Void) {
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .global(qos: .userInteractive))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                self.lock.lock()
                let alreadyFired = self.fired
                self.fired = true
                self.lock.unlock()
                if alreadyFired {
                    _exit(130)
                }
                onStop()
            }
            source.resume()
            sources.append(source)
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

private func loadTranscriber(model: String, verbose: Bool, downloadBasePath: String?) async throws -> VoxtralTranscriber {
    let localModelPath = NSString(string: model).expandingTildeInPath
    if FileManager.default.fileExists(atPath: localModelPath) {
        let modelDirectory = URL(fileURLWithPath: localModelPath).standardizedFileURL
        logVerbose(verbose, "model source: local directory \(modelDirectory.path)")
        let loaded = try VoxtralLoader.load(directory: modelDirectory)
        logVerbose(verbose, "model loaded from: \(loaded.directory.path)")
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    } else {
        let cacheDirectory = VoxtralLoader.resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBasePath)
        logVerbose(verbose, "model source: Hugging Face id \(model)")
        logVerbose(verbose, "hugging face cache root: \(cacheDirectory.path)")
        logVerbose(verbose, "downloading model files if needed...")

        final class ProgressState {
            var lastBucket: Int = -1
        }
        let state = ProgressState()
        let loaded: VoxtralLoadedModel
        do {
            loaded = try await VoxtralLoader.load(
                modelSpec: model,
                downloadBasePath: downloadBasePath
            ) { progress, speed in
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
        } catch VoxtralLoaderError.invalidModelSpec {
            throw ValidationError("Model not found locally and is not a valid Hugging Face repo id: \(model)")
        }
        logVerbose(verbose, "model loaded from cache directory: \(loaded.directory.path)")
        return VoxtralTranscriber(model: loaded.model, tokenizer: loaded.tokenizer, config: loaded.config)
    }
}

private func logVerbose(_ enabled: Bool, _ message: @autoclosure () -> String) {
    guard enabled else { return }
    fputs("[voxmlx] \(message())\n", stderr)
}

private final class MicrophoneChunkSource: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let condition = NSCondition()
    private let chunkSamples: Int
    private var chunkQueue: [[Float]] = []
    private var chunkQueueStart = 0
    private var sampleBuffer: [Float] = []
    private var sampleBufferStart = 0
    private var monoScratch: [Float] = []
    private var resampler: VoxtralAudioResampler?
    private var stopFlag = false
    private var dropped = 0
    private(set) var inputDescription = "unknown"

    init(chunkSamples: Int) throws {
        self.chunkSamples = chunkSamples
    }

    var stopRequested: Bool {
        condition.lock()
        defer { condition.unlock() }
        return stopFlag
    }

    var droppedChunkCount: Int {
        condition.lock()
        defer { condition.unlock() }
        return dropped
    }

    func requestStop() {
        condition.lock()
        stopFlag = true
        condition.broadcast()
        condition.unlock()
    }

    func start() throws {
        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              let tapFormat = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: inputFormat.sampleRate,
                  channels: inputFormat.channelCount,
                  interleaved: false
              )
        else {
            throw MicrophoneChunkSourceError.invalidTapFormat
        }

        let inputRate = Int(inputFormat.sampleRate.rounded())
        resampler = VoxtralAudioResampler(inputRate: inputRate, outputRate: VoxtralAudio.sampleRate)
        inputDescription = "\(inputRate) Hz, \(inputFormat.channelCount) ch -> \(VoxtralAudio.sampleRate) Hz mono"

        // Ask for roughly one model chunk per callback at the input rate.
        let tapFrames = AVAudioFrameCount(
            max(256, Int((Double(chunkSamples) * inputFormat.sampleRate / Double(VoxtralAudio.sampleRate)).rounded()))
        )
        input.installTap(onBus: 0, bufferSize: tapFrames, format: tapFormat) { [weak self] buffer, _ in
            self?.handleBuffer(buffer)
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()

        // Flush the resampler's filter delay into the sample buffer.
        if let resampler {
            let tail = resampler.finish()
            if !tail.isEmpty {
                condition.lock()
                sampleBuffer.append(contentsOf: tail)
                condition.unlock()
            }
        }
    }

    /// Returns the next chunk, waiting up to `timeout` seconds for one to arrive.
    /// Stale chunks beyond `maxBacklogChunks` are dropped so latency stays bounded.
    func popChunk(maxBacklogChunks: Int, timeout: TimeInterval) -> [Float]? {
        condition.lock()
        defer { condition.unlock() }

        let deadline = Date(timeIntervalSinceNow: timeout)
        while chunkQueueStart >= chunkQueue.count, !stopFlag {
            if !condition.wait(until: deadline) {
                break
            }
        }

        guard chunkQueueStart < chunkQueue.count else {
            chunkQueue.removeAll(keepingCapacity: true)
            chunkQueueStart = 0
            return nil
        }

        let backlog = chunkQueue.count - chunkQueueStart
        if backlog > maxBacklogChunks {
            let toDrop = backlog - maxBacklogChunks
            chunkQueueStart += toDrop
            dropped += toDrop
        }
        let chunk = chunkQueue[chunkQueueStart]
        chunkQueueStart += 1
        compactQueueLocked()
        return chunk
    }

    /// Removes and returns every queued chunk (used while shutting down).
    func drainChunks() -> [[Float]] {
        condition.lock()
        defer { condition.unlock() }
        let chunks = Array(chunkQueue[chunkQueueStart...])
        chunkQueue.removeAll(keepingCapacity: true)
        chunkQueueStart = 0
        return chunks
    }

    func consumeRemainder() -> [Float] {
        condition.lock()
        defer { condition.unlock() }
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

    private func compactQueueLocked() {
        if chunkQueueStart == chunkQueue.count {
            chunkQueue.removeAll(keepingCapacity: true)
            chunkQueueStart = 0
        } else if chunkQueueStart > 64, chunkQueueStart * 2 >= chunkQueue.count {
            chunkQueue.removeFirst(chunkQueueStart)
            chunkQueueStart = 0
        }
    }

    private func handleBuffer(_ buffer: AVAudioPCMBuffer) {
        guard buffer.frameLength > 0, let resampler else { return }

        VoxtralAudio.monoSamples(from: buffer, into: &monoScratch)
        let resampled = resampler.process(monoScratch)
        guard !resampled.isEmpty else { return }

        condition.lock()
        sampleBuffer.append(contentsOf: resampled)
        var appended = false
        while sampleBuffer.count - sampleBufferStart >= chunkSamples {
            let start = sampleBufferStart
            let end = start + chunkSamples
            chunkQueue.append(Array(sampleBuffer[start ..< end]))
            sampleBufferStart = end
            appended = true
        }
        if sampleBufferStart == sampleBuffer.count {
            sampleBuffer.removeAll(keepingCapacity: true)
            sampleBufferStart = 0
        } else if sampleBufferStart > 32_768, sampleBufferStart * 2 >= sampleBuffer.count {
            sampleBuffer.removeFirst(sampleBufferStart)
            sampleBufferStart = 0
        }
        if appended {
            condition.signal()
        }
        condition.unlock()
    }
}

private enum MicrophoneChunkSourceError: Error {
    case invalidTapFormat
}
