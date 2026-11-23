#!/usr/bin/env bash
# Mirror the sandbox results dir (+ notebook) into the repo: one file per call, only new/changed files; commit + push.
T="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$T/.." && pwd)"
DEST=$R/results/marimo
listing=$(bash $T/mx.sh - 2>/dev/null <<'PY' | grep '^@@LIST@@' | sed 's/^@@LIST@@//'
import os, json, hashlib
out = {}
for dp, _, fs in os.walk(RESULTS):
    for f in fs:
        if not f.endswith(".pid"):
            p = os.path.join(dp, f); out[os.path.relpath(p, RESULTS)] = hashlib.md5(open(p, "rb").read()).hexdigest()
out["__notebook.py"] = hashlib.md5(open(f"{WORK}/notebook.py", "rb").read()).hexdigest()
print("@@LIST@@" + json.dumps(out))
PY
)
[ -z "$listing" ] && { echo "sync: sandbox unreachable"; exit 1; }
todo=$(python3 -I -c "
import json, sys, hashlib, os
d = json.loads(sys.argv[1])
for k, h in d.items():
    p = '$R/notebooks/pd_defect_experiments.py' if k == '__notebook.py' else '$DEST/' + k
    if not os.path.exists(p) or hashlib.md5(open(p, 'rb').read()).hexdigest() != h:
        print(k)" "$listing")
n_fetch=0
while IFS= read -r k; do
  [ -z "$k" ] && continue
  if [ "$k" = "__notebook.py" ]; then src='f"{WORK}/notebook.py"'; dst=$R/notebooks/pd_defect_experiments.py
  else src="f\"{RESULTS}/$k\""; dst=$DEST/$k; fi
  mkdir -p "$(dirname "$dst")"
  b64=$(bash $T/mx.sh -c "import base64, gzip; print('@@B64@@' + base64.b64encode(gzip.compress(open($src, 'rb').read())).decode())" 2>/dev/null | grep '^@@B64@@' | sed 's/^@@B64@@//')
  if [ -n "$b64" ] && python3 -I -c "import base64,gzip,sys; open(sys.argv[2],'wb').write(gzip.decompress(base64.b64decode(sys.argv[1])))" "$b64" "$dst"; then
    n_fetch=$((n_fetch+1))
  else echo "sync: failed to fetch $k"; fi
done <<< "$todo"
cd $R
git add -A results notebooks
if ! git diff --cached --quiet; then
  n=$(find $DEST -name "res_*.json" | wc -l)
  git commit -q -m "Results sync from marimo sandbox ($n result files)"
  for i in 1 2 3 4; do git -c push.negotiate=true push -q origin main 2>&1 | grep -v '^remote:' && sleep $((2**i)) || break; done
  echo "sync: pushed $n_fetch file(s) ($n result files total)"
else echo "sync: no change"; fi
