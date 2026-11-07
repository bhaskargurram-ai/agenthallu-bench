#!/bin/bash
# Usage: D_run_pilot_parallel.sh MODEL IDX1 IDX2 ...   (one single-task process per index, run concurrently)
MODEL=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd); REAN=$(dirname "$HERE")
mkdir -p "$REAN/D_fabrication/pilot_batches/$MODEL"
for i in "$@"; do
  D=$REAN/D_fabrication/pilot_batches/$MODEL/b$(printf %02d $i)
  if [ -s "$D/incremental.jsonl" ] && grep -q '"status": "ok"' "$D/incremental.jsonl"; then continue; fi
  "$HERE/D_run_pilot_batch.sh" "$MODEL" "$i" $((i+1)) &
done
wait
