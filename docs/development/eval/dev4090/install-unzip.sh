#!/usr/bin/env bash
# Installs an `unzip` compatibility wrapper in the dev4090 WSL user's ~/.local/bin.
# VBench calls `unzip -d DEST FILE` (vbench/utils.py:286,290) with a fixed command name.
# WSL has no unzip and sudo needs a password, so we put a user-level wrapper that
# prefers the Windows-side 7-Zip at /mnt/c/Program Files/7-Zip/7z.exe, with a pure
# Python zipfile fallback. Verified 2026-11-24 on dev4090 (RAFT model unzip OK).
set -uo pipefail
mkdir -p ~/.local/bin

cat > ~/.local/bin/unzip <<'WRAP'
#!/usr/bin/env bash
# unzip compatibility wrapper. Accepts: unzip [-d DEST] FILE [FILE...]
set -e
dest='.'
files=()
while [ $# -gt 0 ]; do
  case "$1" in
    -d) dest="$2"; shift 2 ;;
    -*) shift ;;
    *) files+=("$1"); shift ;;
  esac
done
mkdir -p "$dest"

SEVENZ=''
if command -v 7z >/dev/null 2>&1; then
  SEVENZ="$(command -v 7z)"
elif [ -x '/mnt/c/Program Files/7-Zip/7z.exe' ]; then
  SEVENZ='/mnt/c/Program Files/7-Zip/7z.exe'
fi

if [ -n "$SEVENZ" ]; then
  for f in "${files[@]}"; do
    wf="$f"; wd="$dest"
    case "$SEVENZ" in
      /mnt/c/*)
        if command -v wslpath >/dev/null 2>&1; then
          wf="$(wslpath -w "$f")"; wd="$(wslpath -w "$dest")"
        fi
        ;;
    esac
    "$SEVENZ" x "$wf" -o"$wd" -y >/dev/null 2>&1
  done
else
  python3 - "$dest" "${files[@]}" <<'PY'
import sys, zipfile
dest = sys.argv[1]
for f in sys.argv[2:]:
    with zipfile.ZipFile(f) as z:
        z.extractall(dest)
PY
fi
WRAP
chmod +x ~/.local/bin/unzip
grep -q '\.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "=== wrapper ==="
ls -l ~/.local/bin/unzip
export PATH="$HOME/.local/bin:$PATH"
echo "=== test on existing raft zip ==="
Z="$HOME/.cache/vbench/raft_model/models.zip"
if [ -f "$Z" ]; then
  unzip -d "$HOME/.cache/vbench/raft_model/" "$Z" && echo UNZIP_OK || echo UNZIP_FAIL
else
  echo "NO_ZIP_PRESENT"
fi
