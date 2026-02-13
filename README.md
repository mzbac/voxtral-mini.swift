# Voxtral Mini Swift

Swift port of `mistralai/Voxtral-Mini-4B-Realtime-2602` using `mlx-swift`.

This repo includes:
- A `VoxMLX` Swift library.
- A `VoxMLXCLI` executable for file and realtime microphone transcription.

## Requirements

- Apple Silicon Mac (mlx-swift).
- macOS 14 (Sonoma) or newer.
- Xcode (for `xcodebuild`).
- Network access for first model/audio download.

## Install (precompiled CLI)

You can download a precompiled `VoxMLXCLI` from GitHub Releases (macOS arm64 / Apple Silicon):

- Latest: <https://github.com/mzbac/voxtral-mini.swift/releases/latest>

1. Download `VoxMLXCLI.macos.arm64.zip` from a release.
2. Unzip it and run the CLI from the unzipped folder (the executable expects its `.bundle` resources next to it).

```bash
curl -L -o VoxMLXCLI.macos.arm64.zip https://github.com/mzbac/voxtral-mini.swift/releases/latest/download/VoxMLXCLI.macos.arm64.zip
unzip -o VoxMLXCLI.macos.arm64.zip

chmod +x VoxMLXCLI.macos.arm64/VoxMLXCLI
./VoxMLXCLI.macos.arm64/VoxMLXCLI --help
```

## Build

This repo uses Swift Package Manager, but build/test via `xcodebuild`.

Build and test with local derived data in `./build`:

```bash
xcodebuild -scheme VoxMLX-Package -destination "platform=macOS" -derivedDataPath ./build build
xcodebuild -scheme VoxMLX-Package -destination "platform=macOS" -derivedDataPath ./build test
```

Run the locally built CLI:

```bash
./build/Build/Products/Debug/VoxMLXCLI --help
```

## Model download + cache

`VoxMLXCLI` accepts either a local model directory or a Hugging Face model id via `--model`.

- Local path: a directory containing `tekken.json` and either:
  - original-format files (`params.json` + `.safetensors`)
  - converted-format files (`config.json` + `model*.safetensors`)
- Model id: for example `mistralai/Voxtral-Mini-4B-Realtime-2602` or `mlx-community/Voxtral-Mini-4B-Realtime-6bit`.

When you pass a model id, files are resolved via `swift-transformers` / Hub and cached in the standard Hugging Face cache location (`HF_HUB_CACHE`, or `HF_HOME` + `/hub`, else default cache path).

Authentication uses normal Hugging Face token sources (for example `hf auth login`, `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`).

## CLI

The commands below assume you installed the precompiled release artifact and have `./VoxMLXCLI.macos.arm64/VoxMLXCLI`. If you built from source, replace that path with `./build/Build/Products/Debug/VoxMLXCLI`.

### `transcribe` (file transcription)

```bash
./VoxMLXCLI.macos.arm64/VoxMLXCLI transcribe \
  --audio /path/to/audio.wav \
  --model mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --temp 0 \
  --stats
```

Useful options:
- `--max-new-tokens` to cap generation length.
- `--stats` to print timing and tokens/sec to stderr.

### `live` (realtime microphone)

```bash
./VoxMLXCLI.macos.arm64/VoxMLXCLI live \
  --model mlx-community/Voxtral-Mini-4B-Realtime-6bit \
  --temp 0 \
  --transcription-delay-ms 0 \
  --chunk-ms 80 \
  --decoder-window 2048 \
  --max-backlog-ms 400
```

Stop with `Ctrl+C`.

## Acknowledgements

- `mlx-swift` (Apple MLX): <https://github.com/ml-explore/mlx-swift>
- `swift-transformers` (Hugging Face): <https://github.com/huggingface/swift-transformers>
- `voxmlx` Python reference: <https://github.com/awni/voxmlx>
- `voxtral.c` C reference: <https://github.com/antirez/voxtral.c>
