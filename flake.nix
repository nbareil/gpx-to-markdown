{
  description = "development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    beads.url = "github:steveyegge/beads";
    codex-cli-nix.url = "github:sadjow/codex-cli-nix";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, beads, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        bd = beads.packages.${system}.default.overrideAttrs (old: {
          nativeCheckInputs = (old.nativeCheckInputs or []) ++ [ pkgs.git ];
          doCheck = false; # disable beads Go tests to avoid sandbox failures
        });
        codexCli = inputs."codex-cli-nix".packages.${system}.default;
      in {
        devShells.default = pkgs.mkShell {
          packages =
            (with pkgs; [
              nodejs_20
              nodePackages.pnpm
              watchman
              git
              amp-cli
            ]) ++ [ bd codexCli ];

          shellHook = ''
            export PNPM_HOME=''${PNPM_HOME:-$HOME/.pnpm}
            export PATH=$PNPM_HOME:$PATH
            export NODE_ENV=development
            echo "Loaded dev shell"
          '';
        };
      });
}
