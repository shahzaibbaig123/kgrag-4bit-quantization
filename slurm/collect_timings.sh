#!/bin/bash
# Harvest build/eval wall-clock times from SLURM accounting into a CSV.
# Re-run anytime to refresh; covers all build (f16_*/q4_*) and eval (eval_*) jobs.
#   Usage: ./collect_timings.sh [since_date]   (default: 30 days ago)
SINCE="${1:-$(date -d '30 days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)}"
OUT="build_eval_timings.csv"

echo "kind,model,size,version,jobid,state,elapsed_sec,elapsed_hms,start,end" > "$OUT"

sacct -u "$USER" -S "$SINCE" -X -n -P \
      -o "JobID,JobName,State,ElapsedRaw,Elapsed,Start,End" 2>/dev/null |
while IFS='|' read -r jobid name state esec ehms start end; do
    case "$name" in
        f16_*|q4_*)
            kind=build
            model="${name%%_*}"            # f16 / q4
            size="${name#*_}"              # 200 / 500 / 1000 / 2000 ...
            version="${jobid#*_}"          # array index (= seed); '=jobid' if no array
            [ "$version" = "$jobid" ] && version=""
            ;;
        eval_*)
            kind=eval
            rest="${name#eval_}"           # "f16_500" (new) or "f16" (old, no size)
            model="${rest%%_*}"            # f16 / Q4
            if [ "$rest" != "$model" ]; then size="${rest#*_}"; else size=""; fi
            version="all"                   # eval job covers v0-v4
            ;;
        *) continue ;;
    esac
    echo "$kind,$model,$size,$version,$jobid,$state,$esec,$ehms,$start,$end" >> "$OUT"
done

echo "wrote $OUT ($(($(wc -l < "$OUT")-1)) rows)"
echo "note: eval rows have blank size (job name lacks it); future evals will be named eval_<model>_<size>."
