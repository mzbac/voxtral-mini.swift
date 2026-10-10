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

    /// Loads an audio file, downmixes it to mono and resamples it to ``sampleRate``.
    ///
    /// The file is read in bounded chunks so that memory stays proportional to the
    /// *output* length rather than to the source's sample rate and channel count.
    public static func loadAudio(url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let sourceFormat = audioFile.processingFormat

        guard sourceFormat.commonFormat == .pcmFormatFloat32, sourceFormat.channelCount > 0 else {
            throw VoxtralAudioError.unsupportedAudioFormat(url)
        }

        let readChunkFrames: AVAudioFrameCount = 1 << 16
        guard let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: readChunkFrames) else {
            throw VoxtralAudioError.failedToReadAudio(url)
        }

        let sourceRate = Int(sourceFormat.sampleRate.rounded())
        guard sourceRate > 0 else {
            throw VoxtralAudioError.unsupportedAudioFormat(url)
        }

        let resampler = VoxtralAudioResampler(inputRate: sourceRate, outputRate: sampleRate)
        let expectedOutput = Int((Double(audioFile.length) * Double(sampleRate) / Double(sourceRate)).rounded(.up))

        var output: [Float] = []
        output.reserveCapacity(max(0, expectedOutput) + 1024)
        var mono: [Float] = []

        while audioFile.framePosition < audioFile.length {
            buffer.frameLength = 0
            try audioFile.read(into: buffer, frameCount: readChunkFrames)
            guard buffer.frameLength > 0 else { break }

            monoSamples(from: buffer, into: &mono)
            output.append(contentsOf: resampler.process(mono))
        }
        output.append(contentsOf: resampler.finish())

        return output
    }

    /// Downmixes every channel of `buffer` into a mono signal (channel average).
    public static func monoSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        var out: [Float] = []
        monoSamples(from: buffer, into: &out)
        return out
    }

    /// Downmixes every channel of `buffer` into `out`, reusing its storage.
    public static func monoSamples(from buffer: AVAudioPCMBuffer, into out: inout [Float]) {
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)

        if out.count != frames {
            out = Array(repeating: 0, count: frames)
        }
        guard frames > 0, channels > 0, let channelData = buffer.floatChannelData else {
            return
        }

        out.withUnsafeMutableBufferPointer { dst in
            guard let dstBase = dst.baseAddress else { return }
            let n = vDSP_Length(frames)

            if channels == 1 {
                dstBase.update(from: channelData[0], count: frames)
                return
            }

            vDSP_vclr(dstBase, 1, n)
            if buffer.format.isInterleaved {
                for channel in 0 ..< channels {
                    vDSP_vadd(channelData[0] + channel, vDSP_Stride(channels), dstBase, 1, dstBase, 1, n)
                }
            } else {
                for channel in 0 ..< channels {
                    vDSP_vadd(channelData[channel], 1, dstBase, 1, dstBase, 1, n)
                }
            }

            var scale = 1.0 / Float(channels)
            vDSP_vsmul(dstBase, 1, &scale, dstBase, 1, n)
        }
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
        let combined: [Float]
        if let audioTail {
            combined = audioTail + audioChunk
        } else {
            // First call includes the STFT left padding.
            combined = Array(repeating: Float(0), count: nFFT / 2) + audioChunk
        }

        let (mel, frameCount) = computeLogMel(audio: combined, centerPad: false, dropLastFrame: false)

        // Every frame starts at a multiple of `hopLength`; everything from the first
        // frame start that did not fit is carried over so the next call continues on
        // exactly the same frame grid as the offline spectrogram. A fixed-size tail
        // (nFFT - hopLength) is only correct when the leftover happens to align, and
        // otherwise silently drops samples and shifts every later frame.
        let consumed = frameCount * hopLength
        let newTail = Array(combined[consumed...])

        if frameCount == 0 {
            return (MLXArray.zeros([nMels, 0], dtype: .float32), newTail)
        }
        return (MLXArray(mel).reshaped(nMels, frameCount), newTail)
    }

    /// One-shot resampling of a complete signal to ``sampleRate``.
    ///
    /// For streaming input (e.g. microphone buffers) use ``VoxtralAudioResampler`` so
    /// filter state carries across calls and the fractional phase does not drift.
    public static func resampleToModelRate(_ samples: [Float], inputRate: Int) -> [Float] {
        if inputRate == sampleRate {
            return samples
        }
        let resampler = VoxtralAudioResampler(inputRate: inputRate, outputRate: sampleRate)
        var out = resampler.process(samples)
        out.append(contentsOf: resampler.finish())
        return out
    }

    static func linearResample(samples: [Float], inputRate: Int, outputRate: Int) -> [Float] {
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
        out.replaceSubrange(pad ..< (pad + n), with: samples)

        for i in 0 ..< pad {
            out[pad - 1 - i] = samples[reflectedIndex(1 + i, length: n)]
            out[pad + n + i] = samples[reflectedIndex(n - 2 - i, length: n)]
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

/// Streaming sample-rate converter built on `AVAudioConverter`.
///
/// Unlike per-buffer linear interpolation, the converter applies a proper
/// anti-aliasing filter and keeps its state between calls, so a microphone stream
/// fed in arbitrary buffer sizes neither aliases nor drifts. Call ``finish()`` once
/// at end of stream to flush the filter delay.
public final class VoxtralAudioResampler: @unchecked Sendable {
    public let inputRate: Int
    public let outputRate: Int

    private let converter: AVAudioConverter?
    private let inputFormat: AVAudioFormat?
    private let outputFormat: AVAudioFormat?
    private let lock = NSLock()

    public init(inputRate: Int, outputRate: Int = VoxtralAudio.sampleRate) {
        self.inputRate = inputRate
        self.outputRate = outputRate

        if inputRate != outputRate, inputRate > 0, outputRate > 0,
           let inFormat = AVAudioFormat(
               commonFormat: .pcmFormatFloat32,
               sampleRate: Double(inputRate),
               channels: 1,
               interleaved: false
           ),
           let outFormat = AVAudioFormat(
               commonFormat: .pcmFormatFloat32,
               sampleRate: Double(outputRate),
               channels: 1,
               interleaved: false
           ),
           let converter = AVAudioConverter(from: inFormat, to: outFormat)
        {
            converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
            self.converter = converter
            self.inputFormat = inFormat
            self.outputFormat = outFormat
        } else {
            self.converter = nil
            self.inputFormat = nil
            self.outputFormat = nil
        }
    }

    /// True when no conversion is needed (or possible) and samples pass through unchanged.
    public var isPassthrough: Bool {
        converter == nil
    }

    /// Converts the next block of input samples. Output length may differ from
    /// `samples.count * outputRate / inputRate` for a single call; totals converge.
    public func process(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        guard let converter, let inputFormat, let outputFormat else {
            if inputRate == outputRate || inputRate <= 0 || outputRate <= 0 {
                return samples
            }
            return VoxtralAudio.linearResample(samples: samples, inputRate: inputRate, outputRate: outputRate)
        }

        guard let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = input.floatChannelData
        else {
            return []
        }
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            channel[0].update(from: src.baseAddress!, count: samples.count)
        }

        lock.lock()
        defer { lock.unlock() }
        return drain(converter: converter, outputFormat: outputFormat, input: input, endOfStream: false)
    }

    /// Flushes any samples held in the converter's filter delay and resets it for reuse.
    public func finish() -> [Float] {
        guard let converter, let outputFormat else { return [] }
        lock.lock()
        defer { lock.unlock() }
        let out = drain(converter: converter, outputFormat: outputFormat, input: nil, endOfStream: true)
        converter.reset()
        return out
    }

    private func drain(
        converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        input: AVAudioPCMBuffer?,
        endOfStream: Bool
    ) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let inputFrames = Int(input?.frameLength ?? 0)
        let capacity = AVAudioFrameCount(max(256, Int((Double(inputFrames) * ratio).rounded(.up)) + 256))

        var result: [Float] = []
        var inputDelivered = false

        while true {
            guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { break }

            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, outStatus in
                if let input, !inputDelivered {
                    inputDelivered = true
                    outStatus.pointee = .haveData
                    return input
                }
                outStatus.pointee = endOfStream ? .endOfStream : .noDataNow
                return nil
            }

            if status == .error {
                break
            }

            let produced = Int(output.frameLength)
            if produced > 0, let channel = output.floatChannelData {
                result.append(contentsOf: UnsafeBufferPointer(start: channel[0], count: produced))
            }

            // `.haveData` means the output buffer filled up and more may be pending.
            guard status == .haveData, produced == Int(capacity) else { break }
        }

        return result
    }
}
