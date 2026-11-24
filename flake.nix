{
  description = "Eidolon: Translation Validation for MLIR dialect lowerings with SIMT and SPMD semantics";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];

      # Construct environment for given system
      shellFor = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llvm = pkgs.llvmPackages_23;
        in {
          # Make clang and clang++ the C and C++ compilers
          default = pkgs.mkShell.override { stdenv = llvm.stdenv; } {
            packages = [
              llvm.mlir
              llvm.llvm
              llvm.bintools
              pkgs.z3
              pkgs.cmake
              pkgs.ninja
              pkgs.lit
            ];
          };
        };
    in {
      # Yields devShells.x86_64-linux.default etc which will be opened when `nix develop` is run
      devShells = nixpkgs.lib.genAttrs systems shellFor;
    };
}