{pkgs}: {
  deps = [
    pkgs.ffmpeg
    pkgs.xsimd
    pkgs.libxcrypt
    pkgs.pkg-config
    pkgs.libsndfile
    pkgs.ffmpeg-full
    pkgs.postgresql
    pkgs.openssl
  ];
}
