import Foundation

public struct VoxtralParams: Codable, Sendable {
    public let dim: Int
    public let nLayers: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let ropeTheta: Float
    public let normEps: Float
    public let vocabSize: Int
    public let slidingWindow: Int
    public let multimodal: VoxtralMultimodalConfig
    public let adaRmsNormTCondDim: Int?

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case vocabSize = "vocab_size"
        case slidingWindow = "sliding_window"
        case multimodal
        case adaRmsNormTCondDim = "ada_rms_norm_t_cond_dim"
    }
}

public struct VoxtralMultimodalConfig: Codable, Sendable {
    public let whisperModelArgs: VoxtralWhisperModelArgs

    enum CodingKeys: String, CodingKey {
        case whisperModelArgs = "whisper_model_args"
    }
}

public struct VoxtralWhisperModelArgs: Codable, Sendable {
    public let encoderArgs: VoxtralEncoderArgs
    public let downsampleArgs: VoxtralDownsampleArgs

    enum CodingKeys: String, CodingKey {
        case encoderArgs = "encoder_args"
        case downsampleArgs = "downsample_args"
    }
}

public struct VoxtralDownsampleArgs: Codable, Sendable {
    public let downsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case downsampleFactor = "downsample_factor"
    }
}

public struct VoxtralEncoderArgs: Codable, Sendable {
    public let audioEncodingArgs: VoxtralAudioEncodingArgs
    public let dim: Int
    public let nLayers: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let nHeads: Int
    public let ropeTheta: Float
    public let slidingWindow: Int

    enum CodingKeys: String, CodingKey {
        case audioEncodingArgs = "audio_encoding_args"
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
    }
}

public struct VoxtralAudioEncodingArgs: Codable, Sendable {
    public let samplingRate: Int
    public let frameRate: Float
    public let numMelBins: Int
    public let hopLength: Int
    public let windowSize: Int
    public let globalLogMelMax: Float

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case frameRate = "frame_rate"
        case numMelBins = "num_mel_bins"
        case hopLength = "hop_length"
        case windowSize = "window_size"
        case globalLogMelMax = "global_log_mel_max"
    }
}

extension VoxtralParams {
    public static func load(from url: URL) throws -> VoxtralParams {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VoxtralParams.self, from: data)
    }
}
