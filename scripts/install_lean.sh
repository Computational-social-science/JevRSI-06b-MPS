#!/bin/bash
# Install the Lean toolchain, and do not claim success until Lean actually runs.
#
# The first version of this printed "TOOLCHAIN READY" unconditionally. The
# official download had died mid-stream with an HTTP/2 PROTOCOL_ERROR at 294 MB
# of 562 MB, `tar` had extracted the truncated archive "successfully" enough to
# create bin/lean, and the script reported success while `lean --version` died
# with a missing dylib. A step whose result is never checked fails silently, and
# the message that announces it is the part that makes it convincing.
#
# So: verify the size against the number the publisher states, extract to a
# temporary directory, MOVE it into place only if it works, and run the binary
# before saying anything. A half-installed toolchain in the final location is
# worse than none, because elan will then treat it as present.
set -uo pipefail

TC="$HOME/.elan/toolchains/leanprover--lean4---v4.34.1"
ARCHIVE="${1:-/tmp/lean_gh.tar.zst}"
EXPECT_BYTES="${2:-}"

say() { printf "  %s\n" "$*"; }

if [ ! -f "$ARCHIVE" ]; then
  say "no archive at $ARCHIVE"; exit 1
fi

have=$(stat -f%z "$ARCHIVE")
say "archive: $have bytes ($(($have / 1000000)) MB)"

# 1. SIZE, against the publisher's own number when we have it. A truncated
#    stream is the failure mode that actually happened, and it is invisible to
#    every other check: tar reads a prefix and exits 0.
if [ -n "$EXPECT_BYTES" ] && [ "$have" != "$EXPECT_BYTES" ]; then
  say "SIZE MISMATCH: have $have, publisher says $EXPECT_BYTES — refusing to install"
  exit 2
fi

# 1b. The publisher states a digest, so use it. A size match plus a successful
#     decompression is strong, but "the bytes the publisher intended" is a
#     different and better claim than "a file of the right length opened".
EXPECT_SHA="${3:-}"
if [ -n "$EXPECT_SHA" ]; then
  say "checking sha256 against the publisher's digest..."
  got=$(shasum -a 256 "$ARCHIVE" | cut -d" " -f1)
  say "  got      $got"
  say "  expected $EXPECT_SHA"
  [ "$got" = "$EXPECT_SHA" ] || { say "DIGEST MISMATCH — refusing to install"; exit 7; }
fi

# 2. The archive must actually decompress all the way through. `zstd -t` reads
#    the whole stream, so a truncated download fails here rather than half way
#    through extraction.
if command -v zstd >/dev/null 2>&1; then
  say "testing the zstd stream end to end..."
  zstd -t "$ARCHIVE" 2>&1 | tail -2
  [ "${PIPESTATUS[0]}" -eq 0 ] || { say "archive is truncated or corrupt"; exit 3; }
else
  say "zstd not installed; falling back to extraction into a temp dir and verifying there"
fi

# 3. Extract somewhere disposable. Never into the final location: a partial
#    extraction there is indistinguishable from a good install.
STAGE=$(mktemp -d /tmp/lean-stage.XXXXXX)
trap 'rm -rf "$STAGE" "$STAGE.tar"' EXIT
say "extracting to $STAGE ..."
# Decompress first, then untar. `tar -I zstd` looks the decompressor up on PATH,
# and under launchd and under a non-login shell the Homebrew prefix is often
# absent, so the two-step form fails where the one-step form would have worked
# from an interactive terminal. Two steps, no PATH assumption.
zstd_bin="$(command -v zstd || echo /opt/homebrew/bin/zstd)"
[ -x "$zstd_bin" ] || { say "zstd not found; cannot decompress"; exit 8; }
"$zstd_bin" -d -f -q "$ARCHIVE" -o "$STAGE.tar" || { say "decompress failed"; exit 8; }
tar -xf "$STAGE.tar" -C "$STAGE" --strip-components=1
rm -f "$STAGE.tar"
[ -x "$STAGE/bin/lean" ] || { say "no bin/lean in the archive"; exit 4; }
[ -f "$STAGE/lib/lean/libInit_shared.dylib" ] || {
  say "libInit_shared.dylib missing — the archive is incomplete"; exit 5; }

# 4. THE ACTUAL TEST. Does it run?
if ! out=$("$STAGE/bin/lean" --version 2>&1); then
  say "lean --version FAILED in the staging dir:"; say "$out" | head -5; exit 6
fi
say "staged lean reports: $out"

# 5. Only now does it become the installed toolchain.
rm -rf "$TC"
mkdir -p "$(dirname "$TC")"
mv "$STAGE" "$TC"
trap - EXIT
say "installed to $TC"
"$TC/bin/lean" --version
