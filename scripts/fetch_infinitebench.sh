#!/usr/bin/env bash
# Fetch the InfiniteBench task files used as the long-context prompt corpus.
#
# Run ONCE from a login node (compute nodes have no outbound network). The files
# land outside the repo -- they are hundreds of MB and must not be committed.
#
# Why InfiniteBench: at LIN_TOKENS=120k, chat corpora (ShareGPT ~80 tok/prompt,
# LMSYS-Chat-1M) would need ~1500 conversations stapled together, which
# manufactures a semantic transition every ~80 tokens and makes any routing
# measurement an artifact of the concatenation policy. InfiniteBench records are
# single coherent documents already >100k tokens, so a 120k prompt is one real
# document truncated -- no stapling. It is also ungated (LMSYS-Chat-1M is not).
#
# Tasks chosen for REAL content. The needle-in-a-haystack / synthetic tasks are
# excluded because their context is filler by construction, not because they are
# benchmarks:
#   passkey, number_string  padding is literally "The grass is green. The sky is
#                           blue." repeated -- the exact degenerate pattern this
#                           corpus exists to replace
#   kv_retrieval            UUID pairs
#   math_find, math_calc    bare integer lists / arithmetic expressions
# Accuracy on those tasks is irrelevant here; we use `context` as prompt material
# only. code_run is machine-generated templated functions, but it is real syntax
# rather than filler, so it is included.
set -euo pipefail

BASE="https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main"
DEST="${INFINITEBENCH_DIR:-/scratch/ramanvr_root/ramanvr/akashpt/DeepSeekRun_runtime/corpora/infinitebench}"

# task                   domain       approx tokens in record 0
TASKS=(
  longbook_sum_eng       # literary prose     ~436k
  longbook_choice_eng    # literary prose     ~250k (longest natural-language records)
  longbook_qa_eng        # literary prose     ~95k
  longbook_qa_chn        # Chinese prose      -- different tokenization, different experts
  code_debug             # Python source      ~202k
  longdialogue_qa_eng    # screenplay         ~96k
  code_run               # generated Python   templated but real syntax
)

mkdir -p "$DEST"
echo "[fetch] destination: $DEST"

for t in "${TASKS[@]}"; do
  out="$DEST/$t.jsonl"
  if [[ -s "$out" ]]; then
    echo "[fetch] have   $t.jsonl ($(du -h "$out" | cut -f1))"
    continue
  fi
  echo "[fetch] get    $t.jsonl ..."
  curl -fSL --retry 3 "$BASE/$t.jsonl" -o "$out.part"
  mv "$out.part" "$out"
  echo "[fetch] done   $t.jsonl ($(du -h "$out" | cut -f1))"
done

echo
echo "[fetch] corpus ready:"
du -sh "$DEST"
ls -la "$DEST"
echo
echo "Set PROMPT_CORPUS_PATH to: $DEST"
echo
if [[ -s "$DEST/index.json" ]]; then
  echo "[fetch] index.json present. Re-run the indexer if you added a task above:"
else
  echo "[fetch] NEXT STEP -- the corpus is unusable without an index. Run:"
fi
cat <<EOF
  .venv/bin/python scripts/build_corpus_index.py \\
      --corpus-dir $DEST \\
      --tokenizer-dir <dir containing tokenizer.json>
EOF
echo "(~15 min, one core, login node. Records already indexed are reused.)"
