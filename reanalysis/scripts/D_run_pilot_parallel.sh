#!/bin/bash
# Usage: D_run_pilot_parallel.sh MODEL IDX1 IDX2 ...   (one single-task process per index, run concurrently)
MODEL=$1; shift
mkdir -p /Users/challagullayoshitha/Downloads/agentprop_revision/D_fabrication/pilot_batches/$MODEL
for i in "$@"; do
  D=/Users/challagullayoshitha/Downloads/agentprop_revision/D_fabrication/pilot_batches/$MODEL/b$(printf %02d $i)
  if [ -s "$D/incremental.jsonl" ] && grep -q '"status": "ok"' "$D/incremental.jsonl"; then continue; fi
  /Users/challagullayoshitha/Downloads/agentprop_revision/scripts/D_run_pilot_batch.sh "$MODEL" "$i" $((i+1)) &
done
wait
