#!/bin/bash
# Resumable, HTTP/1.1-forced downloader.
#
# Why this exists, having now failed twice the same way: the Lean toolchain
# tarball is 562 MB, and BOTH the official host and the GitHub release died at
# roughly 295 MB with
#
#     curl: (92) HTTP/2 stream 1 was not closed cleanly: PROTOCOL_ERROR
#
# Two different hosts failing at the same offset is not two coincidences; it is
# something in the path between here and both of them closing an HTTP/2 stream
# after ~300 MB. Retrying the same request just reproduces it, and `--retry`
# alone does not help because the error is a protocol error rather than a
# transient HTTP status.
#
# Two changes, each addressing one half of that:
#
#   --http1.1   no HTTP/2 stream to close badly. The single most likely fix.
#   -C -         resume from the byte we already have, so a connection that dies
#                at 295 MB costs 295 MB of transfer rather than the whole file.
#                The loop repeats until the size matches what the publisher
#                states, so a partial file is a normal intermediate state rather
#                than a failure.
#
# `--retry` is kept but not relied on: the loop is the retry.
set -uo pipefail

URL="$1"
OUT="$2"
EXPECT="${3:-}"
FORCE_HTTP1="${FORCE_HTTP1:-1}"

say() { printf "  %s\n" "$*"; }

attempt=0
stalled=0
prev=0

while :; do
  attempt=$((attempt + 1))
  [ "$attempt" -gt 60 ] && { say "giving up after $attempt attempts"; exit 1; }

  have=0
  [ -f "$OUT" ] && have=$(stat -f%z "$OUT")
  if [ -n "$EXPECT" ] && [ "$have" = "$EXPECT" ]; then
    say "complete: $have bytes"
    exit 0
  fi

  flags=(-sS -L --retry 3 --retry-delay 2 -C - -o "$OUT")
  [ "$FORCE_HTTP1" = "1" ] && flags+=(--http1.1)

  curl "${flags[@]}" "$URL"
  rc=$?

  have=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  say "attempt $attempt: rc=$rc, $(($have / 1000000)) MB / $((${EXPECT:-0} / 1000000)) MB"

  if [ "$have" = "$prev" ]; then
    stalled=$((stalled + 1))
    # Two attempts that added nothing means the resume is not making progress —
    # usually the server ignoring the Range header. Start over once, from zero.
    if [ "$stalled" -ge 2 ] && [ "$have" -gt 0 ] && [ "${FORCE_HTTP1}" = "1" ]; then
      say "no progress twice; restarting from zero"
      rm -f "$OUT"
      stalled=0
    fi
    [ "$stalled" -ge 4 ] && { say "no progress after 4 attempts"; exit 2; }
  else
    stalled=0
  fi
  prev=$have
  sleep 2
done
