#!/usr/bin/env bash
# VBench 6-dimension technical smoke test on dev4090 (RTX 4090).
# Generates 3 synthetic videos, then runs VBench custom_input evaluation on the
# 6 technical dimensions. Results land in ~/vbench-eval/smoke/out/*_eval_results.json.
# Must be run with the SSH session kept alive (nohup/setsid get killed on WSL
# disconnect); see README.md. Verified 2026-11-24.
set -uo pipefail
PY="$HOME/vbench-eval/.venv/bin/python"
DIR="$HOME/vbench-eval/smoke"
mkdir -p "$DIR/videos"
LOG="$HOME/vbench-smoke-final.log"

# WSL gateway proxy: default-route IP + host proxy port 7897.
GW=$(ip route | awk '/^default/ {print $3; exit}')
export http_proxy="http://$GW:7897" https_proxy="http://$GW:7897"
export HTTP_PROXY="http://$GW:7897" HTTPS_PROXY="http://$GW:7897"
export no_proxy="127.0.0.1,localhost"
export PATH="$HOME/.local/bin:$PATH"

echo "=== vbench smoke $(date) proxy=$GW:7897 ===" > "$LOG"
command -v unzip >> "$LOG" 2>&1 || echo "NO unzip" >> "$LOG"

cd "$DIR"

echo "step1 generate 3 test videos" >> "$LOG"
"$PY" <<'PYEOF' >> "$LOG" 2>&1
import numpy as np, imageio.v2 as imageio, os
os.makedirs("videos", exist_ok=True)
rng = np.random.default_rng(0)
for i in range(3):
    w = imageio.get_writer(f"videos/test{i}.mp4", format="FFMPEG", mode="I",
                           fps=24, codec="libx264", quality=8, macro_block_size=16)
    base = rng.random((512, 768, 3)) * 80
    for t in range(125):
        frame = np.clip(base + rng.random((512, 768, 3)) * 40, 0, 255).astype(np.uint8)
        w.append_data(frame)
    w.close()
    print("wrote", f"videos/test{i}.mp4")
PYEOF

echo "step2 verify videos" >> "$LOG"
"$PY" <<'PYEOF' >> "$LOG" 2>&1
import imageio.v2 as imageio, glob
for p in sorted(glob.glob("videos/*.mp4")):
    try:
        r = imageio.get_reader(p, format="FFMPEG")
        n = sum(1 for _ in r); r.close()
        print(p, "frames", n)
    except Exception as e:
        print(p, "FAIL", e)
PYEOF

echo "step3 vbench 6 technical dims" >> "$LOG"
"$PY" -c "
import json, os
from vbench import VBench
dims = ['subject_consistency', 'background_consistency', 'motion_smoothness',
        'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
home = os.path.expanduser('~')
try:
    vb = VBench(device='cuda', full_info_dir=home + '/vbench-eval/VBench_full_info.json',
                output_path=home + '/vbench-eval/smoke/out')
    vb.evaluate(home + '/vbench-eval/smoke/videos', 'smoke_final',
                dimension_list=dims, mode='custom_input')
    # VBench.evaluate() writes results to <output_path>/<name>_eval_results.json and
    # returns None; read the file rather than the return value.
    rp = home + '/vbench-eval/smoke/out/smoke_final_eval_results.json'
    d = json.load(open(rp))
    print('SCORES:', {k: v[0] for k, v in d.items() if isinstance(v, list) and v})
except Exception as e:
    import traceback; traceback.print_exc()
    print('FAILED:', type(e).__name__, str(e)[:1000])
" >> "$LOG" 2>&1
echo "=== done $(date) ===" >> "$LOG"
