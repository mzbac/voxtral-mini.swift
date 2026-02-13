import Foundation
import MLX
import Testing
@testable import VoxMLX

@Test
func tokenizerDecodeIgnoresControlSpecialTokens() throws {
    let json = """
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
      ],
      "audio": {
        "sampling_rate": 16000,
        "frame_rate": 12.5,
        "transcription_delay_ms": 480,
        "streaming_n_left_pad_tokens": 32
      }
    }
    """

    let url = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("tekken-mini-\(UUID().uuidString).json")
    guard let jsonData = json.data(using: .utf8) else {
        struct EncodingError: Error {}
        throw EncodingError()
    }
    try jsonData.write(to: url)

    let tokenizer = try TekkenTokenizer(url: url)
    let text = tokenizer.decode([1, 4, 1, 2, 2], ignoreSpecialTokens: true)

    #expect(tokenizer.bosID == 1)
    #expect(tokenizer.eosID == 2)
    #expect(tokenizer.specialTokenID("[STREAMING_PAD]") == 3)
    #expect(text == "A")
}

@Test
func audioMelShapeLooksReasonable() {
    let durationSeconds: Float = 1.0
    let count = Int(durationSeconds * Float(VoxtralAudio.sampleRate))
    let hz: Float = 440

    var samples = Array(repeating: Float(0), count: count)
    for i in 0 ..< count {
        let t = Float(i) / Float(VoxtralAudio.sampleRate)
        samples[i] = Foundation.sin(2 * Float.pi * hz * t)
    }

    let mel = VoxtralAudio.logMelSpectrogram(samples)
    #expect(mel.dim(0) == VoxtralAudio.nMels)
    #expect(mel.dim(1) > 0)
}

private func makeKV(tokens: [Int32]) -> (MLXArray, MLXArray) {
    let keys = MLXArray(tokens, [1, 1, tokens.count, 1])
    let values = MLXArray(tokens.map { $0 * 10 }, [1, 1, tokens.count, 1])
    return (keys, values)
}

@Test
func rotatingKVCacheSingleTokenSlidingWindowOrder() {
    let cache = RotatingKVCache(maxSize: 4)
    var latest: (MLXArray, MLXArray)?

    for token in 1 ... 5 {
        let (keys, values) = makeKV(tokens: [Int32(token)])
        latest = cache.updateAndFetch(keys: keys, values: values)
    }

    guard let latest else {
        #expect(Bool(false))
        return
    }
    eval(latest.0, latest.1)

    let keyTokens = latest.0.reshaped([-1]).asArray(Int32.self)
    let valueTokens = latest.1.reshaped([-1]).asArray(Int32.self)

    #expect(cache.offset == 5)
    #expect(keyTokens == [5, 2, 3, 4])
    #expect(valueTokens == [50, 20, 30, 40])
}

@Test
func rotatingKVCachePrefillThenDecodeOrder() {
    let cache = RotatingKVCache(maxSize: 4)

    let prefill = makeKV(tokens: [1, 2, 3])
    _ = cache.updateAndFetch(keys: prefill.0, values: prefill.1)

    _ = cache.updateAndFetch(keys: makeKV(tokens: [4]).0, values: makeKV(tokens: [4]).1)
    let latest = cache.updateAndFetch(keys: makeKV(tokens: [5]).0, values: makeKV(tokens: [5]).1)
    eval(latest.0, latest.1)

    let keyTokens = latest.0.reshaped([-1]).asArray(Int32.self)
    let valueTokens = latest.1.reshaped([-1]).asArray(Int32.self)

    #expect(cache.offset == 5)
    #expect(keyTokens == [5, 2, 3, 4])
    #expect(valueTokens == [50, 20, 30, 40])
}
