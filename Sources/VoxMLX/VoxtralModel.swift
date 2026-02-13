import Foundation
import MLX
import MLXFast
import MLXNN

public final class VoxtralTimeEmbedding: @unchecked Sendable {
    private let invFreq: MLXArray

    public init(dim: Int, theta: Float = 10_000.0) {
        let half = max(1, dim / 2)
        var values = Array(repeating: Float(0), count: half)
        for i in 0 ..< half {
            values[i] = Foundation.exp(-Foundation.log(theta) * Float(i) / Float(half))
        }
        self.invFreq = MLXArray(values)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFloat = t.asType(.float32).reshaped(t.size, 1)
        let emb = tFloat * expandedDimensions(invFreq, axis: 0)
        return concatenated([MLX.cos(emb), MLX.sin(emb)], axis: -1)
    }
}

final class VoxtralAudioLanguageAdapter: Module {
    @ModuleInfo(key: "w_in") var wIn: Linear
    @ModuleInfo(key: "w_out") var wOut: Linear

    init(inputDim: Int, outputDim: Int) {
        self._wIn.wrappedValue = Linear(inputDim, outputDim, bias: false)
        self._wOut.wrappedValue = Linear(outputDim, outputDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        wOut(gelu(wIn(x)))
    }
}

final class VoxtralEncoderAttention: Module {
    private let nHeads: Int
    private let headDim: Int
    private let scale: Float
    private let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(dim: Int, nHeads: Int, headDim: Int, ropeTheta: Float) {
        self.nHeads = nHeads
        self.headDim = headDim
        self.scale = Foundation.pow(Float(headDim), -0.5)
        self.ropeTheta = ropeTheta

        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, nHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: RotatingKVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(batch, length, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, nHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(batch, length, nHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = MLXFast.RoPE(
            queries,
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: offset
        )
        keys = MLXFast.RoPE(
            keys,
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: offset
        )

        if let cache {
            let cached = cache.updateAndFetch(keys: keys, values: values)
            keys = cached.0
            values = cached.1
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(out)
    }
}

final class VoxtralEncoderSwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, hiddenDim: Int) {
        self._gateProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class VoxtralEncoderLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralEncoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: VoxtralEncoderSwiGLU

    init(dim: Int, nHeads: Int, headDim: Int, hiddenDim: Int, ropeTheta: Float) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        self._attention.wrappedValue = VoxtralEncoderAttention(
            dim: dim,
            nHeads: nHeads,
            headDim: headDim,
            ropeTheta: ropeTheta
        )
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        self._mlp.wrappedValue = VoxtralEncoderSwiGLU(dim: dim, hiddenDim: hiddenDim)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: RotatingKVCache?
    ) -> MLXArray {
        var out = x + attention(attnNorm(x), mask: mask, cache: cache)
        out = out + mlp(ffnNorm(out))
        return out
    }
}

final class VoxtralWhisperEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    let layers: [VoxtralEncoderLayer]
    let norm: RMSNorm

    init(
        inputChannels: Int,
        dim: Int,
        nLayers: Int,
        nHeads: Int,
        headDim: Int,
        hiddenDim: Int,
        ropeTheta: Float
    ) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: dim,
            kernelSize: 3,
            stride: 1,
            padding: 0,
            bias: true
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 3,
            stride: 2,
            padding: 0,
            bias: true
        )
        self.layers = (0 ..< nLayers).map { _ in
            VoxtralEncoderLayer(
                dim: dim,
                nHeads: nHeads,
                headDim: headDim,
                hiddenDim: hiddenDim,
                ropeTheta: ropeTheta
            )
        }
        self.norm = RMSNorm(dimensions: dim, eps: 1e-5)
    }

    private func causalPad(_ x: MLXArray, amount: Int) -> MLXArray {
        if amount <= 0 { return x }
        return padded(
            x,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((amount, 0)),
                IntOrPair((0, 0)),
            ]
        )
    }

    func forwardConv(_ mel: MLXArray) -> MLXArray {
        var x = expandedDimensions(mel.transposed(1, 0), axis: 0)
        x = gelu(conv1(causalPad(x, amount: 2)))
        x = gelu(conv2(causalPad(x, amount: 1)))
        return x
    }

    func forwardConvStep(
        _ newMel: MLXArray,
        conv1Tail: MLXArray?,
        conv2Tail: MLXArray?
    ) -> (conv2Out: MLXArray, conv1Tail: MLXArray, conv2Tail: MLXArray) {
        // newMel: [1, N, n_mels]
        let conv1Input: MLXArray
        if let conv1Tail {
            conv1Input = concatenated([conv1Tail, newMel], axis: 1)
        } else {
            conv1Input = causalPad(newMel, amount: 2)
        }

        let conv1TailStart = max(0, newMel.dim(1) - 2)
        let newConv1Tail = newMel[0..., conv1TailStart..., 0...]
        let conv1Out = gelu(conv1(conv1Input))

        let conv2Input: MLXArray
        if let conv2Tail {
            conv2Input = concatenated([conv2Tail, conv1Out], axis: 1)
        } else {
            conv2Input = causalPad(conv1Out, amount: 1)
        }

        let conv2TailStart = max(0, conv1Out.dim(1) - 1)
        let newConv2Tail = conv1Out[0..., conv2TailStart..., 0...]
        let conv2Out = gelu(conv2(conv2Input))

        return (conv2Out, newConv1Tail, newConv2Tail)
    }

    func forwardTransformer(_ x: MLXArray, cache: [RotatingKVCache]?) -> MLXArray {
        var out = x
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal
        for (idx, layer) in layers.enumerated() {
            out = layer(out, mask: mask, cache: cache?[idx])
        }
        return norm(out)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let conv = forwardConv(mel)
        return forwardTransformer(conv, cache: nil)
    }
}

final class VoxtralDecoderAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(dim: Int, nHeads: Int, nKVHeads: Int, headDim: Int, ropeTheta: Float) {
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = Foundation.pow(Float(headDim), -0.5)
        self.ropeTheta = ropeTheta

        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: RotatingKVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(batch, length, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(batch, length, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = MLXFast.RoPE(
            queries,
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: offset
        )
        keys = MLXFast.RoPE(
            keys,
            dimensions: headDim,
            traditional: true,
            base: ropeTheta,
            scale: 1.0,
            offset: offset
        )

        if let cache {
            let cached = cache.updateAndFetch(keys: keys, values: values)
            keys = cached.0
            values = cached.1
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(out)
    }
}

final class VoxtralDecoderSwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, hiddenDim: Int) {
        self._gateProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class VoxtralAdaptiveNorm: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, condDim: Int) {
        self._linearIn.wrappedValue = Linear(dim, condDim, bias: false)
        self._linearOut.wrappedValue = Linear(condDim, dim, bias: false)
    }

    func callAsFunction(_ tCond: MLXArray) -> MLXArray {
        linearOut(gelu(linearIn(tCond)))
    }
}

final class VoxtralDecoderLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralDecoderAttention
    @ModuleInfo(key: "ada_norm") var adaNorm: VoxtralAdaptiveNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: VoxtralDecoderSwiGLU

    init(
        dim: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        hiddenDim: Int,
        ropeTheta: Float,
        condDim: Int
    ) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        self._attention.wrappedValue = VoxtralDecoderAttention(
            dim: dim,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            ropeTheta: ropeTheta
        )
        self._adaNorm.wrappedValue = VoxtralAdaptiveNorm(dim: dim, condDim: condDim)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        self._mlp.wrappedValue = VoxtralDecoderSwiGLU(dim: dim, hiddenDim: hiddenDim)
    }

    func callAsFunction(
        _ x: MLXArray,
        tCond: MLXArray,
        adaScale: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: RotatingKVCache?
    ) -> MLXArray {
        let h = attention(attnNorm(x), mask: mask, cache: cache)
        let residual = x + h
        let scale = adaScale ?? ((1.0 as Float) + adaNorm(tCond))
        let ffnIn = ffnNorm(residual) * scale
        return residual + mlp(ffnIn)
    }
}

final class VoxtralLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [VoxtralDecoderLayer]
    let norm: RMSNorm

    init(
        dim: Int,
        nLayers: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        hiddenDim: Int,
        vocabSize: Int,
        ropeTheta: Float,
        condDim: Int
    ) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self.layers = (0 ..< nLayers).map { _ in
            VoxtralDecoderLayer(
                dim: dim,
                nHeads: nHeads,
                nKVHeads: nKVHeads,
                headDim: headDim,
                hiddenDim: hiddenDim,
                ropeTheta: ropeTheta,
                condDim: condDim
            )
        }
        self.norm = RMSNorm(dimensions: dim, eps: 1e-5)
    }

    func embed(_ inputIDs: MLXArray) -> MLXArray {
        embedTokens(inputIDs)
    }

    func makeAdaScales(tCond: MLXArray, dtype: DType) -> [MLXArray] {
        let cond = tCond.asType(dtype)
        return layers.map { (1.0 as Float) + $0.adaNorm(cond) }
    }

    func forwardHidden(
        _ x: MLXArray,
        tCond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [RotatingKVCache]?,
        adaScales: [MLXArray]? = nil
    ) -> MLXArray {
        var out = x
        let cond = adaScales == nil ? tCond.asType(x.dtype) : tCond

        for (idx, layer) in layers.enumerated() {
            out = layer(out, tCond: cond, adaScale: adaScales?[idx], mask: mask, cache: cache?[idx])
        }

        return norm(out)
    }

    func project(_ hidden: MLXArray) -> MLXArray {
        embedTokens.asLinear(hidden)
    }

    func callAsFunction(
        _ x: MLXArray,
        tCond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [RotatingKVCache]?
    ) -> MLXArray {
        project(forwardHidden(x, tCond: tCond, mask: mask, cache: cache, adaScales: nil))
    }
}

public final class VoxtralRealtime: Module {
    public let config: VoxtralParams

    @ModuleInfo(key: "encoder") var encoder: VoxtralWhisperEncoder
    @ModuleInfo(key: "adapter") var adapter: VoxtralAudioLanguageAdapter
    @ModuleInfo(key: "language_model") var languageModel: VoxtralLanguageModel

    public let timeEmbedding: VoxtralTimeEmbedding
    public let downsampleFactor: Int

    public init(_ config: VoxtralParams) {
        self.config = config

        let enc = config.multimodal.whisperModelArgs.encoderArgs
        self.downsampleFactor = config.multimodal.whisperModelArgs.downsampleArgs.downsampleFactor

        self._encoder.wrappedValue = VoxtralWhisperEncoder(
            inputChannels: enc.audioEncodingArgs.numMelBins,
            dim: enc.dim,
            nLayers: enc.nLayers,
            nHeads: enc.nHeads,
            headDim: enc.headDim,
            hiddenDim: enc.hiddenDim,
            ropeTheta: enc.ropeTheta
        )

        self._adapter.wrappedValue = VoxtralAudioLanguageAdapter(
            inputDim: enc.dim * downsampleFactor,
            outputDim: config.dim
        )

        self._languageModel.wrappedValue = VoxtralLanguageModel(
            dim: config.dim,
            nLayers: config.nLayers,
            nHeads: config.nHeads,
            nKVHeads: config.nKVHeads,
            headDim: config.headDim,
            hiddenDim: config.hiddenDim,
            vocabSize: config.vocabSize,
            ropeTheta: config.ropeTheta,
            condDim: config.adaRmsNormTCondDim ?? 32
        )

        self.timeEmbedding = VoxtralTimeEmbedding(dim: config.dim)
    }

    public func makeDecoderCache(maxSize: Int? = nil) -> [RotatingKVCache] {
        let size = maxSize ?? config.slidingWindow
        return (0 ..< config.nLayers).map { _ in RotatingKVCache(maxSize: size) }
    }

    public func makeEncoderCache(maxSize: Int? = nil) -> [RotatingKVCache] {
        let size = maxSize ?? config.multimodal.whisperModelArgs.encoderArgs.slidingWindow
        return (0 ..< config.multimodal.whisperModelArgs.encoderArgs.nLayers).map { _ in
            RotatingKVCache(maxSize: size)
        }
    }

    public func encode(_ mel: MLXArray) -> MLXArray {
        var fixed = mel
        if fixed.dim(1) % 2 != 0 {
            fixed = fixed[0..., 1...]
        }

        let conv = encoder.forwardConv(fixed)
        let seqLength = conv.dim(1)
        guard seqLength > 0 else {
            return MLXArray.zeros([0, config.dim], dtype: conv.dtype)
        }

        let encoderWindow = max(1, config.multimodal.whisperModelArgs.encoderArgs.slidingWindow)
        let chunkSize = max(1, min(256, encoderWindow))
        let cache = makeEncoderCache(maxSize: encoderWindow)

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((seqLength + chunkSize - 1) / chunkSize)

        var start = 0
        while start < seqLength {
            let end = min(seqLength, start + chunkSize)
            let segment = conv[0..., start ..< end, 0...]
            let encodedSegment = encoder.forwardTransformer(segment, cache: cache)
            chunks.append(encodedSegment)
            start = end
        }

        var encoded = concatenated(chunks, axis: 1)
        encoded = encoded.squeezed(axis: 0)

        let remainder = encoded.dim(0) % downsampleFactor
        if remainder != 0 {
            encoded = encoded[remainder..., 0...]
        }

        let groupedLength = encoded.dim(0) / downsampleFactor
        let groupedDim = encoded.dim(1) * downsampleFactor
        let grouped = encoded.reshaped(groupedLength, groupedDim)

        return adapter(grouped)
    }

    public func encodeStep(
        newMel: MLXArray,
        conv1Tail: MLXArray?,
        conv2Tail: MLXArray?,
        encoderCache: [RotatingKVCache]?,
        downsampleBuffer: MLXArray?
    ) -> (
        newAudioEmbeds: MLXArray?,
        conv1Tail: MLXArray?,
        conv2Tail: MLXArray?,
        encoderCache: [RotatingKVCache],
        downsampleBuffer: MLXArray?
    ) {
        let melInput = expandedDimensions(newMel.transposed(1, 0), axis: 0).asType(encoder.conv1.weight.dtype)
        let conv = encoder.forwardConvStep(melInput, conv1Tail: conv1Tail, conv2Tail: conv2Tail)

        let cache = encoderCache ?? makeEncoderCache()
        var encoded = encoder.forwardTransformer(conv.conv2Out, cache: cache)
        encoded = encoded.squeezed(axis: 0)

        if let downsampleBuffer {
            encoded = concatenated([downsampleBuffer, encoded], axis: 0)
        }

        let nComplete = (encoded.dim(0) / downsampleFactor) * downsampleFactor
        guard nComplete > 0 else {
            return (
                nil,
                conv.conv1Tail,
                conv.conv2Tail,
                cache,
                encoded
            )
        }

        let newBuffer: MLXArray?
        if nComplete < encoded.dim(0) {
            newBuffer = encoded[nComplete..., 0...]
        } else {
            newBuffer = nil
        }

        let complete = encoded[0 ..< nComplete, 0...]
        let grouped = complete.reshaped(nComplete / downsampleFactor, -1)
        let audioEmbeds = adapter(grouped)

        return (audioEmbeds, conv.conv1Tail, conv.conv2Tail, cache, newBuffer)
    }

    public func decode(
        _ embeddings: MLXArray,
        tCond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [RotatingKVCache]?
    ) -> MLXArray {
        languageModel(embeddings, tCond: tCond, mask: mask, cache: cache)
    }
}
