#!/bin/bash
# Usage: D_run_pilot_batch.sh MODEL START END  (runs shuffled tasks [START,END) of the 80-task pilot)
MODEL=$1; START=$2; END=$3
KEY=$(grep -E '^OPENAI_API_KEY=' /Users/challagullayoshitha/Downloads/caliche-landman-ai/.env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
export OPENAI_API_KEY="$KEY" TOKENIZERS_PARALLELISM=false
OUT=/Users/challagullayoshitha/Downloads/agentprop_revision/D_fabrication/pilot_batches/$MODEL/b$(printf %02d $START)
cd /Users/challagullayoshitha/Downloads/agenthalbench_complete/revised
python3 paired_pilot_robust.py --model "$MODEL" --start_index $START --n_tasks $END --task_timeout 80 --output_dir "$OUT" > "$OUT.log" 2>&1
echo "$MODEL $START-$END exit=$? lines=$(wc -l < $OUT/incremental.jsonl 2>/dev/null)"
