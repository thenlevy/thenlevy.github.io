{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "hugo-blog";

  buildInputs = with pkgs; [
    hugo
    git
    go
  ];

  shellHook = ''
    echo "Hugo $(hugo version | grep -oP 'v[0-9]+\.[0-9]+\.[0-9]+')" 
  '';
}
