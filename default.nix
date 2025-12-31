{ system ? builtins.currentSystem }:
let
  flake = builtins.getFlake (toString ./.);
  shells = flake.devShells or {};
  hasSystem = builtins.hasAttr system shells;
in if hasSystem then shells.${system}.default else throw "System ${system} not supported by this flake";
