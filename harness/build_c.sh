#!/bin/bash
# Build the two C binaries from ONE source file.
#
# Identical flags, identical source, identical optimisation level. The only difference is
# -DUSE_ACCELERATE, which swaps three nested loops for one cblas_sgemm call. If the two
# binaries post different times, that macro is the only thing that could explain it.
#
# TWO TRAPS ON THIS MACHINE, both cost real time once:
#  1. `cc` is aliased to something else here — always /usr/bin/clang explicitly.
#  2. The default SDK symlink (CommandLineTools/SDKs/MacOSX.sdk -> 27.0) is MALFORMED: its
#     libSystem.B.tbd declares `arm64e.x1-macos`, which ld-1267 rejects with
#     "tapi error: malformed file / unknown architecture". Compiling succeeds and LINKING
#     fails, which reads like a code error and is not one. So we probe for an SDK that
#     actually links instead of trusting the symlink.
set -euo pipefail
cd "$(dirname "$0")/../impls"
mkdir -p ../bin

pick_sdk() {
  local probe="${TMPDIR:-/tmp}/sdkprobe.$$.c" out="${TMPDIR:-/tmp}/sdkprobe.$$"
  printf 'int main(void){return 0;}' > "$probe"
  for sdk in \
    /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
    /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk \
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
    "$(xcrun --show-sdk-path 2>/dev/null || true)"
  do
    [ -d "$sdk" ] || continue
    if /usr/bin/clang -isysroot "$sdk" "$probe" -o "$out" 2>/dev/null; then
      rm -f "$probe" "$out"; echo "$sdk"; return 0
    fi
  done
  rm -f "$probe" "$out"
  echo "no SDK on this machine can link a hello-world; fix Xcode/CommandLineTools" >&2
  return 1
}

SDK="$(pick_sdk)"
echo "sdk: $SDK"
FLAGS=(-O2 -std=c11 -Wall -Wno-unused-result -isysroot "$SDK")

echo "building 03-c-naive      (hand-written matmul)"
/usr/bin/clang "${FLAGS[@]}" -DUSE_ACCELERATE=0 03_04_c.c -o ../bin/03_c_naive -lm

echo "building 04-c-accelerate (cblas_sgemm)"
/usr/bin/clang "${FLAGS[@]}" -DUSE_ACCELERATE=1 03_04_c.c -o ../bin/04_c_accelerate -lm -framework Accelerate

echo "ok:"
ls -la ../bin/
