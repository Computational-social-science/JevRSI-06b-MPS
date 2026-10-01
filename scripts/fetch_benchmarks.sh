#!/usr/bin/env bash
# Fetch the suite's upstream checkouts at the EXACT commits rsijev/targets_suite.py
# pins, and verify each one. A checkout at the wrong commit is worse than none: it
# looks like coverage and silently measures a different split.
#
# Upstream's own rule for why this is a script and not a default: "The loaders read
# a few upstream checkouts from the environment ... None has a default: a default
# would be one author's filesystem, which is how v1.0 shipped a lab path inside a
# checkpoint."
set -u
DEST="${1:?usage: fetch_benchmarks.sh DEST_DIR}"
mkdir -p "$DEST"

ok=0; fail=0; skip=0

resolve_sha() {
  # `git fetch origin <short-sha>` fails on GitHub for an abbreviated sha: the
  # server needs the full object id. Two of the three pins in upstream's own
  # `--list` output are 8 characters, so this is the normal case, not an edge one.
  # A pin that cannot be resolved is reported, never guessed -- fetching a branch
  # tip instead would produce a checkout at the wrong commit, which is worse than
  # no checkout at all, because it looks like coverage.
  #
  # `ls-remote` lists REFS, so it cannot resolve a pin that is not a branch or tag
  # tip -- and jevbench's is not. For those the history has to be walked, which is
  # cheap under `--filter=blob:none --no-checkout`. Unresolvable means the commit
  # was force-pushed away or the repository was rewritten, and that is reported as
  # a finding rather than papered over with a nearby commit.
  local url="$1" want="$2" full
  if [ "${#want}" -eq 40 ]; then echo "$want"; return 0; fi
  full="$(GIT_TERMINAL_PROMPT=0 git ls-remote "$url" 2>/dev/null \
          | awk -v w="$want" '$1 ~ "^"w {print $1; exit}')"
  if [ -n "$full" ]; then echo "$full"; return 0; fi
  local tmp
  tmp="$(mktemp -d)"
  if GIT_TERMINAL_PROMPT=0 git clone -q --filter=blob:none --no-checkout "$url" "$tmp" 2>/dev/null; then
    full="$(git -C "$tmp" rev-parse --verify --quiet "$want^{commit}" 2>/dev/null)"
  fi
  rm -rf "$tmp"
  [ -n "$full" ] && echo "$full" || echo ""
}

fetch_at() {
  local name="$1" url="$2" want="$3" note="${4:-}"
  local sha
  sha="$(resolve_sha "$url" "$want")"
  if [ -z "$sha" ]; then
    printf "  %-10s FAIL %s is not a ref in that repository\n" "$name" "$want"
    fail=$((fail+1)); return
  fi
  local d="$DEST/$name"
  if [ -d "$d/.git" ] && [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    printf "  %-10s already at %s\n" "$name" "${sha:0:8}"; skip=$((skip+1)); return
  fi
  rm -rf "$d"; mkdir -p "$d"
  (
    cd "$d" || exit 1
    git init -q
    git remote add origin "$url"
    GIT_TERMINAL_PROMPT=0 git fetch -q --depth 1 --filter=blob:none origin "$sha"
    git checkout -q FETCH_HEAD
  ) >/dev/null 2>&1
  local got
  got="$(git -C "$d" rev-parse HEAD 2>/dev/null)"
  if [ "$got" = "$sha" ]; then
    local n sz
    n="$(find "$d" -type f -not -path '*/.git/*' | wc -l | tr -d ' ')"
    sz="$(du -sh "$d" 2>/dev/null | cut -f1)"
    printf "  %-10s OK   %s  %s files, %s%s\n" "$name" "${sha:0:8}" "$n" "$sz" "${note:+  ($note)}"
    ok=$((ok+1))
  else
    printf "  %-10s FAIL %s  the commit exists but the tree did not check out\n" \
      "$name" "${sha:0:8}"
    rm -rf "$d"
    fail=$((fail+1))
  fi
}

echo "fetching the suite's pinned checkouts into $DEST"
fetch_at kev      https://github.com/jaredpalmer/kev.git                    569ea449b1033c5aa8f149ec5e20d0c7a296d24a "4 benchmarks, w=0.212"
fetch_at nimble   https://github.com/bespokelabsai/nimble.git               62076b4f                                      "2 benchmarks, w=0.162"
fetch_at jevbench https://github.com/fstandhartinger/jevbench.git           26eb72d4                                      "1 benchmark,  w=0.049"

echo
echo "  ok=$ok  failed=$fail  already=$skip"
[ "$fail" -gt 0 ] && exit 1
exit 0
