import Foundation

public enum TekkenTokenizerError: Error {
    case invalidFile(URL)
    case missingSpecialToken(String)
}

private struct TekkenFile: Decodable {
    struct Config: Decodable {
        let numVocabTokens: Int
        let defaultVocabSize: Int?
        let defaultNumSpecialTokens: Int?

        enum CodingKeys: String, CodingKey {
            case numVocabTokens = "num_vocab_tokens"
            case defaultVocabSize = "default_vocab_size"
            case defaultNumSpecialTokens = "default_num_special_tokens"
        }
    }

    struct VocabEntry: Decodable {
        let rank: Int
        let tokenBytes: String?
        let tokenStr: String?

        enum CodingKeys: String, CodingKey {
            case rank
            case tokenBytes = "token_bytes"
            case tokenStr = "token_str"
        }
    }

    struct SpecialTokenEntry: Decodable {
        let rank: Int
        let tokenStr: String
        let isControl: Bool?

        enum CodingKeys: String, CodingKey {
            case rank
            case tokenStr = "token_str"
            case isControl = "is_control"
        }
    }

    struct Audio: Decodable {
        let samplingRate: Int?
        let frameRate: Float?
        let transcriptionDelayMs: Float?
        let streamingNLeftPadTokens: Int?

        enum CodingKeys: String, CodingKey {
            case samplingRate = "sampling_rate"
            case frameRate = "frame_rate"
            case transcriptionDelayMs = "transcription_delay_ms"
            case streamingNLeftPadTokens = "streaming_n_left_pad_tokens"
        }
    }

    let config: Config
    let vocab: [VocabEntry]
    let specialTokens: [SpecialTokenEntry]
    let audio: Audio?

    enum CodingKeys: String, CodingKey {
        case config
        case vocab
        case specialTokens = "special_tokens"
        case audio
    }
}

private struct TekkenSpecialToken: Sendable {
    let text: String
    let isControl: Bool
}

public struct TekkenAudioMetadata: Sendable {
    public let samplingRate: Int?
    public let frameRate: Float?
    public let transcriptionDelayMs: Float?
    public let streamingNLeftPadTokens: Int?
}

public final class TekkenTokenizer: @unchecked Sendable {
    private let vocabByID: [Data]
    private let specialsByID: [Int: TekkenSpecialToken]
    private let specialIDsByText: [String: Int]
    private let numSpecialTokens: Int

    public let audioMetadata: TekkenAudioMetadata

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let file = try JSONDecoder().decode(TekkenFile.self, from: data)

        let maxSpecialRank = file.specialTokens.map(\.rank).max() ?? -1
        let numSpecial = file.config.defaultNumSpecialTokens ?? max(0, maxSpecialRank + 1)
        let defaultVocabSize = file.config.defaultVocabSize ?? (numSpecial + file.vocab.count)
        let innerVocabSize = max(0, defaultVocabSize - numSpecial)

        var vocab: [Data] = []
        vocab.reserveCapacity(innerVocabSize)

        let sortedVocab = file.vocab.sorted { $0.rank < $1.rank }
        for entry in sortedVocab.prefix(innerVocabSize) {
            if let tokenBytes = entry.tokenBytes,
               let bytes = Data(base64Encoded: tokenBytes)
            {
                vocab.append(bytes)
            } else if let tokenStr = entry.tokenStr {
                vocab.append(Data(tokenStr.utf8))
            } else {
                vocab.append(Data())
            }
        }

        var specialsByID: [Int: TekkenSpecialToken] = [:]
        var specialIDsByText: [String: Int] = [:]

        for id in 0 ..< numSpecial {
            let filler = "<SPECIAL_\(id)>"
            let token = TekkenSpecialToken(text: filler, isControl: true)
            specialsByID[id] = token
            specialIDsByText[filler] = id
        }

        for special in file.specialTokens {
            guard special.rank >= 0, special.rank < numSpecial else { continue }
            let token = TekkenSpecialToken(
                text: special.tokenStr,
                isControl: special.isControl ?? true
            )
            specialsByID[special.rank] = token
            specialIDsByText[special.tokenStr] = special.rank
        }

        self.vocabByID = vocab
        self.specialsByID = specialsByID
        self.specialIDsByText = specialIDsByText
        self.numSpecialTokens = numSpecial
        self.audioMetadata = TekkenAudioMetadata(
            samplingRate: file.audio?.samplingRate,
            frameRate: file.audio?.frameRate,
            transcriptionDelayMs: file.audio?.transcriptionDelayMs,
            streamingNLeftPadTokens: file.audio?.streamingNLeftPadTokens
        )
    }

    public var bosID: Int {
        specialTokenID("<s>") ?? 1
    }

    public var eosID: Int {
        specialTokenID("</s>") ?? 2
    }

    public var unkID: Int {
        specialTokenID("<unk>") ?? 0
    }

    public func specialTokenID(_ token: String) -> Int? {
        specialIDsByText[token]
    }

    func decodedBytes(for tokenID: Int, ignoreSpecialTokens: Bool = true) -> Data {
        if tokenID >= 0, tokenID < numSpecialTokens, let special = specialsByID[tokenID] {
            if ignoreSpecialTokens || special.isControl {
                return Data()
            }
            return Data(special.text.utf8)
        }

        let vocabID = tokenID - numSpecialTokens
        guard vocabID >= 0, vocabID < vocabByID.count else {
            return Data()
        }
        return vocabByID[vocabID]
    }

    public func decode(_ tokenIDs: [Int], ignoreSpecialTokens: Bool = true) -> String {
        var bytes = Data()
        bytes.reserveCapacity(tokenIDs.count * 2)

        for tokenID in tokenIDs {
            bytes.append(decodedBytes(for: tokenID, ignoreSpecialTokens: ignoreSpecialTokens))
        }

        return String(decoding: bytes, as: UTF8.self)
    }
}
