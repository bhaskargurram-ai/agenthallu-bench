#!/bin/bash
# Usage: D_run_pilot_batch.sh MODEL START END  (runs shuffled tasks [START,END) of the 80-task pilot)
# Requires OPENAI_API_KEY in the environment.
MODEL=$1; START=$2; END=$3
HERE=$(cd "$(dirname "$0")" && pwd); REAN=$(dirname "$HERE")
export TOKENIZERS_PARALLELISM=false
OUT=$REAN/D_fabrication/pilot_batches/$MODEL/b$(printf %02d $START)
cd "$REAN/pilot"
python3 paired_pilot_robust.py --model "$MODEL" --start_index $START --n_tasks $END --task_timeout 80 --output_dir "$OUT" > "$OUT.log" 2>&1
echo "$MODEL $START-$END exit=$? lines=$(wc -l < $OUT/incremental.jsonl 2>/dev/null)"
