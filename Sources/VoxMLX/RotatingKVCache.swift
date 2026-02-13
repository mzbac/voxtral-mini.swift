import MLX

public final class RotatingKVCache: @unchecked Sendable {
    private static let step = 256

    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    public private(set) var offset: Int = 0
    private var idx: Int = 0

    public let maxSize: Int

    public init(maxSize: Int) {
        self.maxSize = maxSize
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
        idx = 0
    }

    private func trim(_ trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        let trimmed = trimSize > 0 ? array[0..., 0..., trimSize..., 0...] : array

        if let append {
            return concatenated([trimmed, append], axis: 2)
        }
        return trimmed
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[0..., 0..., idx..., 0...],
                    array[0..., 0..., ..<idx, 0...],
                ],
                axis: 2
            )
        } else {
            return array[0..., 0..., ..<idx, 0...]
        }
    }

    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if keys == nil || values == nil {
            keys = newKeys
            values = newValues
        } else {
            keys = temporalOrder(keys!)
            values = temporalOrder(values!)
            idx = keys!.dim(2)

            let trimSize = idx + newKeys.dim(2) - maxSize
            keys = trim(trimSize, keys!, append: newKeys)
            values = trim(trimSize, values!, append: newValues)
        }

        offset += newKeys.dim(2)
        idx = keys!.dim(2)
        return (keys!, values!)
    }

    private func updateInPlace(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let batch = newKeys.dim(0)
        let nKVHeads = newKeys.dim(1)
        let step = newKeys.dim(2)
        let kHeadDim = newKeys.dim(3)
        let vHeadDim = newValues.dim(3)
        let previousOffset = offset

        if keys == nil
            || (previousOffset >= keys!.dim(2) && keys!.dim(2) < maxSize)
        {
            let newSize = min(Self.step, maxSize - previousOffset)
            let newKeysStorage = MLXArray.zeros(
                [batch, nKVHeads, newSize, kHeadDim],
                dtype: newKeys.dtype
            )
            let newValuesStorage = MLXArray.zeros(
                [batch, nKVHeads, newSize, vHeadDim],
                dtype: newValues.dtype
            )

            if let existingKeys = keys, let existingValues = values {
                keys = concatenated([existingKeys, newKeysStorage], axis: 2)
                values = concatenated([existingValues, newValuesStorage], axis: 2)
            } else {
                keys = newKeysStorage
                values = newValuesStorage
            }
            idx = previousOffset
        }

        let trimSize = (keys?.dim(2) ?? 0) - maxSize
        if trimSize > 0, let existingKeys = keys, let existingValues = values {
            keys = trim(trimSize, existingKeys)
            values = trim(trimSize, existingValues)
            idx = maxSize
        }

        if idx == maxSize {
            idx = 0
        }

        if let cachedKeys = keys, let cachedValues = values {
            cachedKeys[0..., 0..., idx ..< (idx + step), 0...] = newKeys
            cachedValues[0..., 0..., idx ..< (idx + step), 0...] = newValues
            keys = cachedKeys
            values = cachedValues
        }

        offset += step
        idx += step

        guard let cachedKeys = keys, let cachedValues = values else {
            return (newKeys, newValues)
        }

        if offset < maxSize {
            return (
                cachedKeys[0..., 0..., ..<offset, 0...],
                cachedValues[0..., 0..., ..<offset, 0...]
            )
        }
        return (cachedKeys, cachedValues)
    }

    public func updateAndFetch(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if newKeys.dim(2) == 1 {
            return updateInPlace(keys: newKeys, values: newValues)
        }
        return updateConcat(keys: newKeys, values: newValues)
    }
}

@available(*, deprecated, renamed: "RotatingKVCache")
public typealias SimpleKVCache = RotatingKVCache
