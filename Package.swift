// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "VoxMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VoxMLX", targets: ["VoxMLX"]),
        .executable(name: "VoxMLXCLI", targets: ["VoxMLXCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.1.6")
        ),
        .package(
            url: "https://github.com/apple/swift-argument-parser.git",
            .upToNextMinor(from: "1.4.0")
        ),
    ],
    targets: [
        .target(
            name: "VoxMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/VoxMLX"
        ),
        .executableTarget(
            name: "VoxMLXCLI",
            dependencies: [
                "VoxMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/VoxMLXCLI"
        ),
        .testTarget(
            name: "VoxMLXTests",
            dependencies: ["VoxMLX"],
            path: "Tests/VoxMLXTests"
        ),
    ]
)
