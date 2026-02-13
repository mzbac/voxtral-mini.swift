import Foundation
import Testing
@testable import VoxMLX

private enum ObamaE2EError: Error {
    case missingAudioURL
    case invalidAudioURL
}

private enum ObamaE2EConfig {
    static let runFlagPath = "/tmp/VOXMLX_RUN_E2E"
    static let modelPathOverrideFile = "/tmp/VOXMLX_MODEL_ID"
    static let maxTokensOverrideFile = "/tmp/VOXMLX_E2E_MAX_NEW_TOKENS"

    static func shouldRun() -> Bool {
        let env = ProcessInfo.processInfo.environment
        if env["VOXMLX_RUN_E2E"] == "1" {
            return true
        }
        let flagPath = env["VOXMLX_E2E_FLAG_FILE"] ?? runFlagPath
        return FileManager.default.fileExists(atPath: flagPath)
    }

    static func modelID() -> String {
        let env = ProcessInfo.processInfo.environment
        if let value = env["VOXMLX_MODEL_ID"], !value.isEmpty {
            return value
        }
        if let data = FileManager.default.contents(atPath: modelPathOverrideFile),
           let value = String(data: data, encoding: .utf8)?
               .trimmingCharacters(in: .whitespacesAndNewlines),
           !value.isEmpty
        {
            return value
        }
        return VoxtralLoader.defaultModelID
    }

    static func maxNewTokens(default value: Int) -> Int {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["VOXMLX_E2E_MAX_NEW_TOKENS"], let parsed = Int(raw) {
            return parsed
        }
        if let data = FileManager.default.contents(atPath: maxTokensOverrideFile),
           let raw = String(data: data, encoding: .utf8)?
               .trimmingCharacters(in: .whitespacesAndNewlines),
           let parsed = Int(raw)
        {
            return parsed
        }
        return value
    }
}

private struct ObamaDatasetRowsResponse: Decodable {
    struct RowContainer: Decodable {
        struct RowData: Decodable {
            struct AudioEntry: Decodable {
                let src: String
                let type: String?
            }

            let audio: [AudioEntry]
        }

        let row: RowData
    }

    let rows: [RowContainer]
}

@Test
func obamaVoiceDatasetEndToEnd() async throws {
    guard ObamaE2EConfig.shouldRun() else {
        return
    }

    let datasetRowsURL = URL(
        string: "https://datasets-server.huggingface.co/rows?dataset=RaysDipesh%2Fobama-voice-samples-283&config=default&split=train&offset=0&length=1"
    )!

    let (rowsData, _) = try await URLSession.shared.data(from: datasetRowsURL)
    let payload = try JSONDecoder().decode(ObamaDatasetRowsResponse.self, from: rowsData)
    guard let src = payload.rows.first?.row.audio.first?.src else {
        throw ObamaE2EError.missingAudioURL
    }
    guard let audioURL = URL(string: src) else {
        throw ObamaE2EError.invalidAudioURL
    }

    let (audioData, _) = try await URLSession.shared.data(from: audioURL)
    let localAudioURL = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("obama-e2e-\(UUID().uuidString).wav")
    try audioData.write(to: localAudioURL)

    let modelID = ObamaE2EConfig.modelID()
    let maxNewTokens = ObamaE2EConfig.maxNewTokens(default: 160)

    let transcriber: VoxtralTranscriber
    if FileManager.default.fileExists(atPath: modelID) {
        transcriber = try VoxtralTranscriber.load(from: URL(fileURLWithPath: modelID))
    } else {
        transcriber = try await VoxtralTranscriber.load(modelID: modelID)
    }
    let text = try transcriber.transcribe(audioURL: localAudioURL, temperature: 0, maxNewTokens: maxNewTokens)

    let normalized = text.lowercased()
    #expect(normalized.contains("three years ago in our state of wisconsin"))
    #expect(normalized.contains("sikh temple"))
}
