import Accelerate
import AVFoundation
import Foundation
import MLX

public enum VoxtralAudioError: Error {
    case unsupportedAudioFormat(URL)
    case failedToReadAudio(URL)
}

public enum VoxtralAudio {
    public static let sampleRate = 16_000
    public static let nFFT = 400
    public static let hopLength = 160
    public static let nMels = 128
    public static let globalLogMelMax: Float = 1.5
    public static let samplesPerToken = hopLength * 2 * 4  // hop * conv stride * downsample

    public static func loadAudio(url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let sourceFormat = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sourceFormat.sampleRate,
            channels: sourceFormat.channelCount,
            interleaved: false
        ) else {
            throw VoxtralAudioError.unsupportedAudioFormat(url)
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw VoxtralAudioError.failedToReadAudio(url)
        }

        try audioFile.read(into: buffer)

        let channels = Int(format.channelCount)
        let frames = Int(buffer.frameLength)

        guard channels > 0, frames > 0, let channelData = buffer.floatChannelData else {
            throw VoxtralAudioError.failedToReadAudio(url)
        }

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
            let inv = 1.0 / Float(channels)
            for i in 0 ..< frames {
                mono[i] *= inv
            }
        }

        let sr = Int(round(format.sampleRate))
        if sr == sampleRate {
            return mono
        }
        return linearResample(samples: mono, inputRate: sr, outputRate: sampleRate)
    }

    public static func padAudio(
        _ audio: [Float],
        leftPadTokens: Int = 32,
        rightPadTokens: Int = 17
    ) -> [Float] {
        let leftPad = leftPadTokens * samplesPerToken
        let rightAlign = (samplesPerToken - (audio.count % samplesPerToken)) % samplesPerToken
        let rightPad = rightAlign + rightPadTokens * samplesPerToken

        var out = Array(repeating: Float(0), count: leftPad)
        out.reserveCapacity(leftPad + audio.count + rightPad)
        out.append(contentsOf: audio)
        out.append(contentsOf: repeatElement(Float(0), count: rightPad))
        return out
    }

    public static func logMelSpectrogram(_ audio: [Float]) -> MLXArray {
        let (mel, frameCount) = computeLogMel(audio: audio, centerPad: true, dropLastFrame: true)
        if frameCount == 0 {
            return MLXArray.zeros([nMels, 0], dtype: .float32)
        }
        return MLXArray(mel).reshaped(nMels, frameCount)
    }

    public static func logMelSpectrogramStep(
        audioChunk: [Float],
        audioTail: [Float]?
    ) -> (mel: MLXArray, newTail: [Float]) {
        let tailLen = nFFT - hopLength  // 240

        let combined: [Float]
        if let audioTail {
            combined = audioTail + audioChunk
        } else {
            // First call includes STFT left padding.
            combined = Array(repeating: Float(0), count: nFFT / 2) + audioChunk
        }

        let newTail: [Float]
        if combined.count <= tailLen {
            newTail = combined
        } else {
            newTail = Array(combined.suffix(tailLen))
        }

        let (mel, frameCount) = computeLogMel(audio: combined, centerPad: false, dropLastFrame: false)
        if frameCount == 0 {
            return (MLXArray.zeros([nMels, 0], dtype: .float32), newTail)
        }
        return (MLXArray(mel).reshaped(nMels, frameCount), newTail)
    }

    public static func resampleToModelRate(_ samples: [Float], inputRate: Int) -> [Float] {
        if inputRate == sampleRate {
            return samples
        }
        return linearResample(samples: samples, inputRate: inputRate, outputRate: sampleRate)
    }

    private static func linearResample(samples: [Float], inputRate: Int, outputRate: Int) -> [Float] {
        guard !samples.isEmpty, inputRate > 0, outputRate > 0 else { return [] }
        if inputRate == outputRate { return samples }

        let outCount = max(1, Int((Int64(samples.count) * Int64(outputRate)) / Int64(inputRate)))
        if outCount == 1 {
            return [samples[0]]
        }

        var out = Array(repeating: Float(0), count: outCount)
        for i in 0 ..< outCount {
            let srcPos = (Float(i) * Float(inputRate)) / Float(outputRate)
            let idx = Int(srcPos)
            let frac = srcPos - Float(idx)

            if idx + 1 < samples.count {
                out[i] = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            } else if idx < samples.count {
                out[i] = samples[idx]
            } else {
                out[i] = 0
            }
        }

        return out
    }

    private static func computeLogMel(
        audio: [Float],
        centerPad: Bool,
        dropLastFrame: Bool
    ) -> ([Float], Int) {
        var signal = audio
        if centerPad {
            let pad = nFFT / 2
            signal = reflectPad(signal, pad: pad)
        }

        guard signal.count >= nFFT else {
            return ([], 0)
        }

        let totalFrames = 1 + (signal.count - nFFT) / hopLength
        let frameCount = dropLastFrame ? max(0, totalFrames - 1) : totalFrames
        guard frameCount > 0 else {
            return ([], 0)
        }

        let window = SpectrogramCache.window
        let dftReal = SpectrogramCache.dftReal
        let dftImag = SpectrogramCache.dftImag
        let melFilter = SpectrogramCache.melFilterBank
        let nFreqs = nFFT / 2 + 1

        var melOut = Array(repeating: Float(0), count: nMels * frameCount)
        var frame = Array(repeating: Float(0), count: nFFT)
        var real = Array(repeating: Float(0), count: nFreqs)
        var imag = Array(repeating: Float(0), count: nFreqs)
        var power = Array(repeating: Float(0), count: nFreqs)
        var imagSquared = Array(repeating: Float(0), count: nFreqs)
        var melEnergy = Array(repeating: Float(0), count: nMels)

        for frameIndex in 0 ..< frameCount {
            let start = frameIndex * hopLength
            for i in 0 ..< nFFT {
                frame[i] = signal[start + i] * window[i]
            }

            sgemvRowMajor(matrix: dftReal, rows: nFreqs, cols: nFFT, vector: frame, result: &real)
            sgemvRowMajor(matrix: dftImag, rows: nFreqs, cols: nFFT, vector: frame, result: &imag)

            vDSP.multiply(real, real, result: &power)
            vDSP.multiply(imag, imag, result: &imagSquared)
            vDSP.add(power, imagSquared, result: &power)

            sgemvRowMajor(matrix: melFilter, rows: nMels, cols: nFreqs, vector: power, result: &melEnergy)

            for m in 0 ..< nMels {
                var logSpec = Foundation.log10(max(melEnergy[m], 1e-10))
                logSpec = max(logSpec, globalLogMelMax - 8.0)
                logSpec = (logSpec + 4.0) / 4.0

                melOut[m * frameCount + frameIndex] = logSpec
            }
        }

        return (melOut, frameCount)
    }

    private static func reflectPad(_ samples: [Float], pad: Int) -> [Float] {
        guard pad > 0 else { return samples }
        guard !samples.isEmpty else { return Array(repeating: 0, count: pad * 2) }
        guard samples.count > 1 else {
            return Array(repeating: samples[0], count: samples.count + pad * 2)
        }

        let n = samples.count
        var out = Array(repeating: Float(0), count: n + pad * 2)

        for outIndex in out.indices {
            let sourceIndex = reflectedIndex(outIndex - pad, length: n)
            out[outIndex] = samples[sourceIndex]
        }

        return out
    }

    private static func reflectedIndex(_ index: Int, length: Int) -> Int {
        precondition(length > 1)
        let maxIndex = length - 1
        var i = index

        while i < 0 || i > maxIndex {
            if i < 0 {
                i = -i
            } else {
                i = 2 * maxIndex - i
            }
        }

        return i
    }

    private static func sgemvRowMajor(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: [Float],
        result: inout [Float]
    ) {
        precondition(vector.count == cols)
        precondition(result.count == rows)

        matrix.withUnsafeBufferPointer { matrixPtr in
            vector.withUnsafeBufferPointer { vectorPtr in
                result.withUnsafeMutableBufferPointer { resultPtr in
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(rows),
                        Int32(cols),
                        1.0,
                        matrixPtr.baseAddress!,
                        Int32(cols),
                        vectorPtr.baseAddress!,
                        1,
                        0.0,
                        resultPtr.baseAddress!,
                        1
                    )
                }
            }
        }
    }
}

private enum SpectrogramCache {
    static let window: [Float] = {
        // Hann(N + 1)[:-1], matching the Python implementation.
        let n = VoxtralAudio.nFFT
        return (0 ..< n).map { i in
            let x = 2.0 * Double.pi * Double(i) / Double(n)
            return Float(0.5 - 0.5 * Foundation.cos(x))
        }
    }()

    static let dftReal: [Float] = {
        let nFFT = VoxtralAudio.nFFT
        let nFreqs = nFFT / 2 + 1
        var out = Array(repeating: Float(0), count: nFreqs * nFFT)
        for k in 0 ..< nFreqs {
            for n in 0 ..< nFFT {
                let angle = -2.0 * Double.pi * Double(k * n) / Double(nFFT)
                out[k * nFFT + n] = Float(Foundation.cos(angle))
            }
        }
        return out
    }()

    static let dftImag: [Float] = {
        let nFFT = VoxtralAudio.nFFT
        let nFreqs = nFFT / 2 + 1
        var out = Array(repeating: Float(0), count: nFreqs * nFFT)
        for k in 0 ..< nFreqs {
            for n in 0 ..< nFFT {
                let angle = -2.0 * Double.pi * Double(k * n) / Double(nFFT)
                out[k * nFFT + n] = Float(Foundation.sin(angle))
            }
        }
        return out
    }()

    static let melFilterBank: [Float] = {
        melFilterBank(
            sampleRate: VoxtralAudio.sampleRate,
            nFFT: VoxtralAudio.nFFT,
            nMels: VoxtralAudio.nMels,
            fMin: 0,
            fMax: 8000
        )
    }()

    private static func hzToMel(_ frequency: Double) -> Double {
        let minLogHz = 1000.0
        let minLogMel = 15.0
        let logStep = 27.0 / Foundation.log(6.4)

        if frequency >= minLogHz {
            return minLogMel + Foundation.log(frequency / minLogHz) * logStep
        }
        return 3.0 * frequency / 200.0
    }

    private static func melToHz(_ mel: Double) -> Double {
        let minLogHz = 1000.0
        let minLogMel = 15.0
        let logStep = Foundation.log(6.4) / 27.0

        if mel >= minLogMel {
            return minLogHz * Foundation.exp(logStep * (mel - minLogMel))
        }
        return 200.0 * mel / 3.0
    }

    private static func melFilterBank(
        sampleRate: Int,
        nFFT: Int,
        nMels: Int,
        fMin: Double,
        fMax: Double
    ) -> [Float] {
        let nFreqs = nFFT / 2 + 1

        let nyquist = Double(sampleRate) / 2.0
        var fftFreqs: [Double] = []
        fftFreqs.reserveCapacity(nFreqs)
        for i in 0 ..< nFreqs {
            let frequency = Double(i) * nyquist / Double(nFreqs - 1)
            fftFreqs.append(frequency)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melFreqs: [Double] = []
        melFreqs.reserveCapacity(nMels + 2)
        let melDelta = melMax - melMin
        let melDenominator = Double(nMels + 1)
        for i in 0 ..< (nMels + 2) {
            let mel = melMin + melDelta * Double(i) / melDenominator
            melFreqs.append(mel)
        }

        var filterFreqs: [Double] = []
        filterFreqs.reserveCapacity(melFreqs.count)
        for mel in melFreqs {
            filterFreqs.append(melToHz(mel))
        }

        var filters = Array(repeating: Float(0), count: nMels * nFreqs)

        for m in 0 ..< nMels {
            let left = filterFreqs[m]
            let center = filterFreqs[m + 1]
            let right = filterFreqs[m + 2]
            let norm = 2.0 / max(right - left, 1e-12)

            for (k, frequency) in fftFreqs.enumerated() {
                let down = (frequency - left) / max(center - left, 1e-12)
                let up = (right - frequency) / max(right - center, 1e-12)
                let triangular = max(0.0, min(down, up))
                filters[m * nFreqs + k] = Float(triangular * norm)
            }
        }

        return filters
    }
}
