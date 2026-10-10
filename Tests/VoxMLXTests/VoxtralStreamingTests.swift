import AVFoundation
import Foundation
import MLX
import Testing
@testable import VoxMLX

// MARK: - Helpers

private func testSignal(seconds: Float, sampleRate: Int = VoxtralAudio.sampleRate, seed: UInt64 = 7) -> [Float] {
    let count = Int(seconds * Float(sampleRate))
    var state = seed
    var out = Array(repeating: Float(0), count: count)
    for i in 0 ..< count {
        // Deterministic LCG noise mixed with two tones so every mel bin has energy.
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let noise = Float(Int64(bitPattern: state >> 11) % 20_001) / 10_000.0 - 1.0
        let t = Float(i) / Float(sampleRate)
        out[i] = 0.3 * Foundation.sin(2 * Float.pi * 220 * t)
            + 0.2 * Foundation.sin(2 * Float.pi * 3_100 * t)
            + 0.05 * noise
    }
    return out
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    eval(a, b)
    return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private func tinyParams() throws -> VoxtralParams {
    let json = """
    {
      "dim": 32, "n_layers": 2, "head_dim": 8, "hidden_dim": 64, "n_heads": 4, "n_kv_heads": 2,
      "rope_theta": 10000.0, "norm_eps": 1e-5, "vocab_size": 64, "sliding_window": 4096,
      "ada_rms_norm_t_cond_dim": 8,
      "multimodal": {
        "whisper_model_args": {
          "encoder_args": {
            "audio_encoding_args": {
              "sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
              "hop_length": 160, "window_size": 400, "global_log_mel_max": 1.5
            },
            "dim": 16, "n_layers": 2, "head_dim": 8, "hidden_dim": 32, "n_heads": 2,
            "rope_theta": 10000.0, "sliding_window": 750
          },
          "downsample_args": { "downsample_factor": 4 }
        }
      }
    }
    """
    return try JSONDecoder().decode(VoxtralParams.self, from: Data(json.utf8))
}

/// MLX evaluates lazily on a shared Metal device and is not safe to drive from several
/// threads at once. swift-testing runs tests in parallel by default, so every test
/// that touches MLX lives in this serialized suite.
@Suite(.serialized)
struct MLXTests {
    init() {
        MLXTestDevice.configure()
    }
}

/// Selects the MLX backend for the suite from `VOXMLX_TEST_DEVICE` (`cpu` or `gpu`).
///
/// GitHub's hosted macOS runners expose a paravirtualized GPU that reports a
/// tensor-core capable architecture but whose Metal compiler cannot build MLX's
/// `steel_gemm_fused_nax` kernels ("unsupported deferred-static-alloca-size function
/// body"). The first matmul then aborts the whole test process. CI sets the variable
/// to `cpu`; locally the GPU remains the default.
private enum MLXTestDevice {
    private static let configured: Void = {
        let requested = ProcessInfo.processInfo.environment["VOXMLX_TEST_DEVICE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch requested {
        case "cpu":
            Device.setDefault(device: Device(.cpu))
        case "gpu":
            Device.setDefault(device: Device(.gpu))
        default:
            break
        }
    }()

    static func configure() {
        _ = configured
    }
}

// MARK: - Mel framing

extension MLXTests {

@Test
func streamingMelMatchesOfflineFrameGrid() {
    // Leading silence makes the offline reflect padding and the streaming zero
    // padding coincide, exactly as the realtime session's left pad does.
    let audio = Array(repeating: Float(0), count: 2 * VoxtralAudio.samplesPerToken) + testSignal(seconds: 1.5)
    let offline = VoxtralAudio.logMelSpectrogram(audio)

    // Mixed chunk sizes exercise every leftover alignment, including chunks too
    // short to produce a frame.
    let chunkSizes = [VoxtralAudio.samplesPerToken, 100, 700, 1280, 1280, 333, 2560]
    var tail: [Float]?
    var streamed: [MLXArray] = []
    var cursor = 0
    var chunkIndex = 0
    while cursor < audio.count {
        let size = min(chunkSizes[chunkIndex % chunkSizes.count], audio.count - cursor)
        let step = VoxtralAudio.logMelSpectrogramStep(audioChunk: Array(audio[cursor ..< (cursor + size)]), audioTail: tail)
        tail = step.newTail
        if step.mel.dim(1) > 0 {
            streamed.append(step.mel)
        }
        cursor += size
        chunkIndex += 1
    }

    let stream = concatenated(streamed, axis: 1)
    let frames = stream.dim(1)
    // Streaming never produces frames that would need right padding, so it trails
    // the offline spectrogram by at most a couple of frames and never leads it.
    #expect(frames > 0)
    #expect(frames <= offline.dim(1))
    #expect(offline.dim(1) - frames <= 2)

    let diff = maxAbsDiff(offline[0..., 0 ..< frames], stream)
    #expect(diff < 1e-4, "streaming mel frames are shifted relative to offline: max diff \(diff)")
}

// MARK: - Encoder front-end

@Test
func streamingConvStepMatchesOfflineConv() {
    let encoder = VoxtralWhisperEncoder(
        inputChannels: 8, dim: 16, nLayers: 1, nHeads: 2, headDim: 8, hiddenDim: 32, ropeTheta: 10_000
    )
    let totalFrames = 41
    let mel = MLXRandom.normal([8, totalFrames])  // [n_mels, frames], as logMelSpectrogram returns
    let offline = encoder.forwardConv(mel)  // [1, T', dim]

    // Odd and even chunk lengths, including a chunk shorter than the conv kernel.
    let chunkLengths = [7, 8, 1, 2, 9, 8, 6]
    var conv1Tail: MLXArray?
    var conv2Tail: MLXArray?
    var pieces: [MLXArray] = []
    var start = 0
    var i = 0
    while start < totalFrames {
        let length = min(chunkLengths[i % chunkLengths.count], totalFrames - start)
        let chunk = expandedDimensions(mel[0..., start ..< (start + length)].transposed(1, 0), axis: 0)
        let step = encoder.forwardConvStep(chunk, conv1Tail: conv1Tail, conv2Tail: conv2Tail)
        conv1Tail = step.conv1Tail
        conv2Tail = step.conv2Tail
        if step.conv2Out.dim(1) > 0 {
            pieces.append(step.conv2Out)
        }
        start += length
        i += 1
    }

    let streamed = concatenated(pieces, axis: 1)
    #expect(streamed.dim(1) == offline.dim(1))
    let diff = maxAbsDiff(offline, streamed)
    #expect(diff < 1e-4, "streaming conv output drifted from offline: max diff \(diff)")
}

@Test
func streamingEncodeMatchesOfflineEncode() throws {
    let params = try tinyParams()
    let model = VoxtralRealtime(params)
    eval(model)

    let totalFrames = 96  // 12 audio tokens' worth of mel frames (8 per token)
    let mel = MLXRandom.normal([128, totalFrames])
    let offline = model.encode(mel)  // [tokens, dim]

    var conv1Tail: MLXArray?
    var conv2Tail: MLXArray?
    var cache: [RotatingKVCache]?
    var downsampleBuffer: MLXArray?
    var embeds: [MLXArray] = []

    let chunkLengths = [8, 8, 7, 9, 8, 3, 5, 8]
    var start = 0
    var i = 0
    while start < totalFrames {
        let length = min(chunkLengths[i % chunkLengths.count], totalFrames - start)
        let step = model.encodeStep(
            newMel: mel[0..., start ..< (start + length)],
            conv1Tail: conv1Tail,
            conv2Tail: conv2Tail,
            encoderCache: cache,
            downsampleBuffer: downsampleBuffer
        )
        conv1Tail = step.conv1Tail
        conv2Tail = step.conv2Tail
        cache = step.encoderCache
        downsampleBuffer = step.downsampleBuffer
        if let newEmbeds = step.newAudioEmbeds {
            embeds.append(newEmbeds)
        }
        start += length
        i += 1
    }

    let streamed = concatenated(embeds, axis: 0)
    #expect(streamed.dim(0) == offline.dim(0))
    #expect(streamed.dim(1) == offline.dim(1))
    let diff = maxAbsDiff(offline, streamed)
    #expect(diff < 1e-3, "streaming encoder output drifted from offline: max diff \(diff)")
}

}

// MARK: - Resampler

@Test
func streamingResamplerKeepsLengthAndLevel() {
    let inputRate = 48_000
    let input = testSignal(seconds: 2.0, sampleRate: inputRate)
    let resampler = VoxtralAudioResampler(inputRate: inputRate)
    #expect(!resampler.isPassthrough)

    var output: [Float] = []
    var cursor = 0
    let chunkSizes = [1_000, 4_800, 123, 2_048]
    var i = 0
    while cursor < input.count {
        let size = min(chunkSizes[i % chunkSizes.count], input.count - cursor)
        output.append(contentsOf: resampler.process(Array(input[cursor ..< (cursor + size)])))
        cursor += size
        i += 1
    }
    output.append(contentsOf: resampler.finish())

    let expected = input.count * VoxtralAudio.sampleRate / inputRate
    #expect(abs(output.count - expected) <= 64, "got \(output.count) samples, expected ~\(expected)")

    func rms(_ x: ArraySlice<Float>) -> Float {
        Foundation.sqrt(x.reduce(0) { $0 + $1 * $1 } / Float(max(1, x.count)))
    }
    // Compare steady-state level (skip filter ramp-up at both ends). The 3.1 kHz tone
    // and the noise survive the 8 kHz cutoff, so levels should be close.
    let inLevel = rms(input[(inputRate / 4) ..< (input.count - inputRate / 4)])
    let outLevel = rms(output[(VoxtralAudio.sampleRate / 4) ..< (output.count - VoxtralAudio.sampleRate / 4)])
    #expect(abs(inLevel - outLevel) / inLevel < 0.15, "level changed from \(inLevel) to \(outLevel)")

    let passthrough = VoxtralAudioResampler(inputRate: VoxtralAudio.sampleRate)
    #expect(passthrough.isPassthrough)
    #expect(passthrough.process([1, 2, 3]) == [1, 2, 3])
}

@Test
func loadAudioResamplesAndDownmixes() throws {
    let rate = 44_100.0
    guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2) else {
        Issue.record("could not create format")
        return
    }
    let frames = Int(rate * 0.5)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)),
          let data = buffer.floatChannelData
    else {
        Issue.record("could not create buffer")
        return
    }
    buffer.frameLength = AVAudioFrameCount(frames)
    for i in 0 ..< frames {
        let t = Float(i) / Float(rate)
        data[0][i] = 0.5 * Foundation.sin(2 * Float.pi * 440 * t)
        data[1][i] = -data[0][i]  // channels cancel exactly: mono mix must be ~0
    }

    let url = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("voxmlx-load-\(UUID().uuidString).wav")
    // Scope the writer so the file is flushed and closed before it is read back.
    try {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }()
    defer { try? FileManager.default.removeItem(at: url) }

    let loaded = try VoxtralAudio.loadAudio(url: url)
    let expected = Int(0.5 * Double(VoxtralAudio.sampleRate))
    let peak = loaded.map { abs($0) }.max() ?? 1
    #expect(abs(loaded.count - expected) <= 64, "loaded \(loaded.count) samples, expected ~\(expected)")
    #expect(peak < 1e-4, "mono mix of cancelling channels peaked at \(peak)")
}

// MARK: - UTF-8 streaming

@Test
func utf8IncompleteSuffixDetection() {
    func held(_ bytes: [UInt8]) -> Int {
        UTF8StreamDecoder.incompleteSuffixLength(of: Data(bytes))
    }
    #expect(held([]) == 0)
    #expect(held(Array("abc".utf8)) == 0)
    #expect(held(Array("é".utf8)) == 0)
    #expect(held([0xC3]) == 1)  // first byte of "é"
    #expect(held(Array("a".utf8) + [0xE2, 0x82]) == 2)  // 2 of 3 bytes of "€"
    #expect(held([0xF0, 0x9F, 0x98]) == 3)  // 3 of 4 bytes of an emoji
    #expect(held(Array("€".utf8)) == 0)
    #expect(held([0xFF]) == 0)  // never valid, must not be held back
    #expect(held([0x80, 0x80, 0x80, 0x80]) == 0)  // stray continuations
}

// MARK: - Weight loading

private let miniTekkenJSON = """
{
  "config": { "num_vocab_tokens": 8 },
  "vocab": [
    { "rank": 4, "token_bytes": "QQ==" },
    { "rank": 5, "token_bytes": "Qg==" },
    { "rank": 6, "token_bytes": "Qw==" }
  ],
  "special_tokens": [
    { "rank": 0, "token_str": "<unk>", "is_control": true },
    { "rank": 1, "token_str": "<s>", "is_control": true },
    { "rank": 2, "token_str": "</s>", "is_control": true },
    { "rank": 3, "token_str": "[STREAMING_PAD]", "is_control": true }
  ]
}
"""

private func writeTinyModelDirectory(dropping droppedKey: String?) throws -> URL {
    let params = try tinyParams()
    let model = VoxtralRealtime(params)
    eval(model)

    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("voxmlx-tiny-model-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    try Data(miniTekkenJSON.utf8).write(to: dir.appendingPathComponent("tekken.json"))
    try JSONEncoder().encode(params).write(to: dir.appendingPathComponent("config.json"))

    var weights: [String: MLXArray] = [:]
    for (key, value) in model.parameters().flattened() {
        weights[key] = value
    }
    #expect(weights.count > 10)
    if let droppedKey {
        #expect(weights.removeValue(forKey: droppedKey) != nil, "test expects \(droppedKey) to exist")
    }
    try save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
    return dir
}

extension MLXTests {

@Test
func loaderAcceptsCompleteCheckpoint() throws {
    let dir = try writeTinyModelDirectory(dropping: nil)
    defer { try? FileManager.default.removeItem(at: dir) }

    let loaded = try VoxtralLoader.load(directory: dir)
    #expect(loaded.config.dim == 32)
    #expect(loaded.tokenizer.eosID == 2)
}

@Test
func loaderRejectsCheckpointWithMissingWeights() throws {
    // A checkpoint missing a tensor must fail loudly instead of silently running with
    // a randomly initialised layer.
    let dir = try writeTinyModelDirectory(dropping: "language_model.layers.1.mlp.down_proj.weight")
    defer { try? FileManager.default.removeItem(at: dir) }

    #expect(throws: (any Error).self) {
        try VoxtralLoader.load(directory: dir)
    }
}

}
