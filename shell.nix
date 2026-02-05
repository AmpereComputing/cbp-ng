{ pkgs ? import (builtins.fetchTarball {
    name = "nixpkgs_25.11_2026-01-26";
    url = "https://github.com/NixOS/nixpkgs/archive/1cd347bf3355fce6c64ab37d3967b4a2cb4b878c.tar.gz";
    sha256 = "195pkrjal51mr7v6psjvx1ap3vcnyvp076kzngjfg4cgvskplg1j";
  }) {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs.buildPackages; [
    clang
    clang-tools
    cmake
    gawk
    gcc
    gdb
    git
    parallel
    python3
    libcxx
    zlib
  ];
}
