import Foundation
import Hub
import MLX
import MLXNN

public enum VoxtralLoaderError: Error {
    case missingFile(URL)
    case unsupportedModelFormat(URL)
    case invalidModelSpec(String)
}

private struct VoxtralQuantizationConfig: Decodable {
    let groupSize: Int
    let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct VoxtralConfigMetadata: Decodable {
    let quantization: VoxtralQuantizationConfig?
}

public struct VoxtralLoadedModel {
    public let model: VoxtralRealtime
    public let tokenizer: TekkenTokenizer
    public let config: VoxtralParams
    public let directory: URL
}

public enum VoxtralLoader {
    public static let defaultModelID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    public static let defaultRealtimeModelID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"

    private static let downloadGlobs = [
        "consolidated.safetensors",
        "model*.safetensors",
        "model.safetensors.index.json",
        "params.json",
        "config.json",
        "tekken.json",
    ]

    public static func resolveHuggingFaceHubCacheDirectory(downloadBasePath: String? = nil) -> URL {
        if let downloadBasePath = normalizedNonEmpty(downloadBasePath) {
            return normalizedPathURL(downloadBasePath)
        }

        let env = ProcessInfo.processInfo.environment
        if let hubCache = normalizedNonEmpty(env["HF_HUB_CACHE"]) {
            return normalizedPathURL(hubCache)
        }
        if let hfHome = normalizedNonEmpty(env["HF_HOME"]) {
            return normalizedPathURL(hfHome).appendingPathComponent("hub").standardizedFileURL
        }

        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .standardizedFileURL
    }

    public static func downloadModel(
        modelID: String,
        revision: String = "main",
        hubApi: HubApi? = nil,
        progressHandler: ((_ progress: Progress, _ speedBytesPerSec: Double?) -> Void)? = nil
    ) async throws -> URL {
        // Avoid false offline mode during short-lived CLI/test runs.
        setenv("CI_DISABLE_NETWORK_MONITOR", "1", 0)

        let hubApi = hubApi ?? makeDefaultHubApi(downloadBasePath: nil)

        return try await hubApi.snapshot(
            from: modelID,
            revision: revision,
            matching: downloadGlobs,
            progressHandler: { progress, speed in
                progressHandler?(progress, speed)
            }
        )
    }

    public static func load(
        modelID: String,
        revision: String = "main",
        hubApi: HubApi? = nil,
        progressHandler: ((_ progress: Progress, _ speedBytesPerSec: Double?) -> Void)? = nil
    ) async throws -> VoxtralLoadedModel {
        let directory = try await downloadModel(
            modelID: modelID,
            revision: revision,
            hubApi: hubApi,
            progressHandler: progressHandler
        )
        return try load(directory: directory)
    }

    public static func load(
        modelSpec: String,
        defaultRevision: String = "main",
        downloadBasePath: String? = nil,
        progressHandler: ((_ progress: Progress, _ speedBytesPerSec: Double?) -> Void)? = nil
    ) async throws -> VoxtralLoadedModel {
        let localDirectory = normalizedPathURL(modelSpec)
        if FileManager.default.fileExists(atPath: localDirectory.path) {
            return try load(directory: localDirectory)
        }

        let (repoID, revision) = parseModelSpec(modelSpec, defaultRevision: defaultRevision)
        guard isLikelyHuggingFaceModelID(repoID) else {
            throw VoxtralLoaderError.invalidModelSpec(modelSpec)
        }

        let hubCacheDir = resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBasePath)
        if let cachedSnapshot = findCachedSnapshot(repoID: repoID, revision: revision, hubCacheDir: hubCacheDir) {
            return try load(directory: cachedSnapshot)
        }

        let hubApi = makeDefaultHubApi(downloadBasePath: hubCacheDir.path)
        let directory = try await downloadModel(
            modelID: repoID,
            revision: revision,
            hubApi: hubApi,
            progressHandler: progressHandler
        )
        return try load(directory: directory)
    }

    public static func load(directory: URL) throws -> VoxtralLoadedModel {
        let paramsURL = directory.appendingPathComponent("params.json")
        let configURL = directory.appendingPathComponent("config.json")
        let tekkenURL = directory.appendingPathComponent("tekken.json")

        guard FileManager.default.fileExists(atPath: tekkenURL.path) else {
            throw VoxtralLoaderError.missingFile(tekkenURL)
        }

        let tokenizer = try TekkenTokenizer(url: tekkenURL)

        if FileManager.default.fileExists(atPath: paramsURL.path) {
            let config = try VoxtralParams.load(from: paramsURL)
            let model = VoxtralRealtime(config)

            let raw = try loadSafetensors(in: directory)
            let mapped = remapOriginalWeights(raw)
            try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .none)
            eval(model)

            return VoxtralLoadedModel(
                model: model,
                tokenizer: tokenizer,
                config: config,
                directory: directory
            )
        }

        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            let metadata = try JSONDecoder().decode(VoxtralConfigMetadata.self, from: data)
            let config = try JSONDecoder().decode(VoxtralParams.self, from: data)
            let model = VoxtralRealtime(config)

            if let quantization = metadata.quantization {
                quantize(
                    model: model,
                    filter: { _, module in
                        if let linear = module as? Linear {
                            if linear.weight.dim(-1) % quantization.groupSize == 0 {
                                return (quantization.groupSize, quantization.bits, .affine)
                            }
                            return nil
                        }

                        if let embedding = module as? Embedding {
                            if embedding.weight.dim(-1) % quantization.groupSize == 0 {
                                return (quantization.groupSize, quantization.bits, .affine)
                            }
                            return nil
                        }

                        return nil
                    }
                )
            }

            let raw = try loadSafetensors(in: directory)
            try model.update(parameters: ModuleParameters.unflattened(raw), verify: .none)
            eval(model)

            return VoxtralLoadedModel(
                model: model,
                tokenizer: tokenizer,
                config: config,
                directory: directory
            )
        }

        throw VoxtralLoaderError.unsupportedModelFormat(directory)
    }

    private static func makeDefaultHubApi(downloadBasePath: String?) -> HubApi {
        let base = resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBasePath)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return HubApi(downloadBase: base)
    }

    private static func parseModelSpec(_ modelSpec: String, defaultRevision: String) -> (repoID: String, revision: String) {
        let parts = modelSpec.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let repoID = String(parts[0])
        if parts.count == 2 {
            let revision = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (repoID, revision.isEmpty ? defaultRevision : revision)
        }
        return (repoID, defaultRevision)
    }

    private static func findCachedSnapshot(repoID: String, revision: String, hubCacheDir: URL) -> URL? {
        let swiftPath = hubCacheDir
            .appendingPathComponent("models")
            .appendingPathComponent(repoID)
            .standardizedFileURL
        if isValidSnapshotDirectory(swiftPath) {
            return swiftPath
        }

        let modelIDKey = repoID.replacingOccurrences(of: "/", with: "--")
        let pythonBase = hubCacheDir
            .appendingPathComponent("models--\(modelIDKey)")
            .standardizedFileURL
        if let pythonSnapshot = resolvePythonSnapshot(base: pythonBase, revision: revision),
           isValidSnapshotDirectory(pythonSnapshot) {
            return pythonSnapshot
        }

        return nil
    }

    private static func resolvePythonSnapshot(base: URL, revision: String) -> URL? {
        let fm = FileManager.default
        let snapshotsDir = base.appendingPathComponent("snapshots")
        guard fm.fileExists(atPath: snapshotsDir.path) else {
            return nil
        }

        let refPath = base.appendingPathComponent("refs").appendingPathComponent(revision)
        if let data = try? Data(contentsOf: refPath),
           let raw = String(data: data, encoding: .utf8)?
           .trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            let revisionSnapshot = snapshotsDir.appendingPathComponent(raw)
            if isValidSnapshotDirectory(revisionSnapshot) {
                return revisionSnapshot
            }
        }

        guard let snapshots = try? fm.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        let sorted = snapshots.sorted { lhs, rhs in
            let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lhsDate > rhsDate
        }
        return sorted.first(where: isValidSnapshotDirectory(_:))
    }

    private static func isValidSnapshotDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.path) else {
            return false
        }

        let tekkenURL = directory.appendingPathComponent("tekken.json")
        guard fm.fileExists(atPath: tekkenURL.path) else {
            return false
        }

        let paramsURL = directory.appendingPathComponent("params.json")
        let configURL = directory.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: paramsURL.path) || fm.fileExists(atPath: configURL.path) else {
            return false
        }

        return directoryContainsSafetensors(directory)
    }

    private static func directoryContainsSafetensors(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return false
        }
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    private static func isLikelyHuggingFaceModelID(_ modelID: String) -> Bool {
        if modelID.hasPrefix("/") || modelID.hasPrefix("./") || modelID.hasPrefix("../") || modelID.hasPrefix("~/") {
            return false
        }

        let parts = modelID.split(separator: "/")
        guard parts.count == 2 else {
            return false
        }

        let org = String(parts[0])
        let repo = String(parts[1])
        guard !org.isEmpty && !repo.isEmpty else {
            return false
        }

        let pathIndicators = [
            "models", "model", "weights", "data", "datasets", "checkpoints", "output",
            "tmp", "temp", "cache",
        ]
        if pathIndicators.contains(org.lowercased()) {
            return false
        }

        if org.filter({ $0 == "." }).count > 1 {
            return false
        }

        return true
    }

    private static func normalizedNonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private static func normalizedPathURL(_ path: String) -> URL {
        URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
    }

    private static func loadSafetensors(in directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !files.isEmpty else {
            throw VoxtralLoaderError.unsupportedModelFormat(directory)
        }

        var weights: [String: MLXArray] = [:]
        for file in files {
            let loaded = try loadArrays(url: file, stream: .cpu)
            for (name, tensor) in loaded {
                weights[name] = tensor
            }
        }
        return weights
    }

    private static func remapOriginalWeights(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(raw.count)

        for (rawName, rawTensor) in raw {
            if rawName == "output.weight" {
                continue
            }

            guard let mappedName = remapName(rawName) else {
                continue
            }

            var tensor = rawTensor
            if isConvWeight(mappedName) {
                tensor = tensor.transposed(0, 2, 1)
            }
            mapped[mappedName] = tensor
        }

        return mapped
    }

    private static func remapName(_ name: String) -> String? {
        var stripped = name
        if stripped.hasPrefix("mm_streams_embeddings.embedding_module.") {
            stripped.removeFirst("mm_streams_embeddings.embedding_module.".count)
        }
        if stripped.hasPrefix("mm_whisper_embeddings.") {
            stripped.removeFirst("mm_whisper_embeddings.".count)
        }

        for pattern in remapPatterns {
            if let remapped = pattern.apply(to: stripped) {
                return remapped
            }
        }

        return nil
    }

    private static func isConvWeight(_ name: String) -> Bool {
        (name.contains("conv1.weight") || name.contains("conv2.weight")) && !name.contains("bias")
    }

    private struct RegexRemap {
        let regex: NSRegularExpression
        let replacement: String

        init(_ pattern: String, _ replacement: String) {
            self.regex = try! NSRegularExpression(pattern: "^" + pattern + "$")
            self.replacement = replacement
        }

        func apply(to input: String) -> String? {
            let range = NSRange(input.startIndex..<input.endIndex, in: input)
            guard regex.firstMatch(in: input, range: range) != nil else {
                return nil
            }
            return regex.stringByReplacingMatches(in: input, range: range, withTemplate: replacement)
        }
    }

    private static let remapPatterns: [RegexRemap] = [
        // Encoder conv layers
        RegexRemap(#"whisper_encoder\.conv_layers\.0\.conv\.(.*)"#, "encoder.conv1.$1"),
        RegexRemap(#"whisper_encoder\.conv_layers\.1\.conv\.(.*)"#, "encoder.conv2.$1"),

        // Encoder transformer
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq\.(.*)"#, "encoder.layers.$1.attention.q_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk\.(.*)"#, "encoder.layers.$1.attention.k_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv\.(.*)"#, "encoder.layers.$1.attention.v_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(.*)"#, "encoder.layers.$1.attention.o_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(.*)"#, "encoder.layers.$1.attn_norm.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(.*)"#, "encoder.layers.$1.mlp.gate_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(.*)"#, "encoder.layers.$1.mlp.down_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(.*)"#, "encoder.layers.$1.mlp.up_proj.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(.*)"#, "encoder.layers.$1.ffn_norm.$2"),
        RegexRemap(#"whisper_encoder\.transformer\.norm\.(.*)"#, "encoder.norm.$1"),

        // Adapter
        RegexRemap(#"audio_language_projection\.0\.weight"#, "adapter.w_in.weight"),
        RegexRemap(#"audio_language_projection\.2\.weight"#, "adapter.w_out.weight"),

        // Language model
        RegexRemap(#"tok_embeddings\.weight"#, "language_model.embed_tokens.weight"),
        RegexRemap(#"layers\.(\d+)\.attention\.wq\.weight"#, "language_model.layers.$1.attention.q_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.attention\.wk\.weight"#, "language_model.layers.$1.attention.k_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.attention\.wv\.weight"#, "language_model.layers.$1.attention.v_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.attention\.wo\.weight"#, "language_model.layers.$1.attention.o_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.attention_norm\.weight"#, "language_model.layers.$1.attn_norm.weight"),
        RegexRemap(#"layers\.(\d+)\.feed_forward\.w1\.weight"#, "language_model.layers.$1.mlp.gate_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.feed_forward\.w2\.weight"#, "language_model.layers.$1.mlp.down_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.feed_forward\.w3\.weight"#, "language_model.layers.$1.mlp.up_proj.weight"),
        RegexRemap(#"layers\.(\d+)\.ffn_norm\.weight"#, "language_model.layers.$1.ffn_norm.weight"),
        RegexRemap(#"layers\.(\d+)\.ada_rms_norm_t_cond\.0\.weight"#, "language_model.layers.$1.ada_norm.linear_in.weight"),
        RegexRemap(#"layers\.(\d+)\.ada_rms_norm_t_cond\.2\.weight"#, "language_model.layers.$1.ada_norm.linear_out.weight"),
        RegexRemap(#"norm\.weight"#, "language_model.norm.weight"),
    ]
}
