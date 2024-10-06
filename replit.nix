{pkgs}: {
  deps = [
    pkgs.libsndfile
    pkgs.ffmpeg-full
    pkgs.postgresql
    pkgs.openssl
  ];
}
