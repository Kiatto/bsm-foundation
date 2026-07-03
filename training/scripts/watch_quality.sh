#!/bin/bash
# Monitor quality training and auto-run evaluation when complete.
# Usage: bash training/scripts/watch_quality.sh
# This will poll every 60s and run eval when the BLMF file appears.

MODEL="checkpoints/tinystories_quality.blmf"
QUALITY_DIR="checkpoints/quality"
EVAL_SCRIPT="training/scripts/evaluate_quality.sh"
LOG="/tmp/quality_watch.log"

echo "[$(date)] Watching for quality training to complete..."
echo "  Model target: $MODEL"
echo "  Log: $LOG"
echo ""

while true; do
    if [ -f "$MODEL" ]; then
        MODEL_SIZE=$(stat -c%s "$MODEL" 2>/dev/null)
        echo "[$(date)] Model found! Size: $MODEL_SIZE bytes"
        
        # Wait a moment to ensure export is complete
        sleep 10
        
        # Verify model is stable (size not changing)
        SIZE1=$(stat -c%s "$MODEL" 2>/dev/null)
        sleep 5
        SIZE2=$(stat -c%s "$MODEL" 2>/dev/null)
        
        if [ "$SIZE1" = "$SIZE2" ] && [ "$SIZE1" -gt 0 ]; then
            echo "[$(date)] Model stable. Starting evaluation pipeline..."
            echo "[$(date)] === QUALITY TRAINING COMPLETE ===" >> "$LOG"
            
            bash "$EVAL_SCRIPT" 2>&1 | tee -a "$LOG"
            echo "[$(date)] Evaluation complete!" | tee -a "$LOG"
            break
        else
            echo "[$(date)] Model size changed ($SIZE1 -> $SIZE2), waiting..."
        fi
    else
        # Show latest checkpoint
        LATEST=$(ls -t "$QUALITY_DIR"/checkpoint_step_*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            STEP=$(echo "$LATEST" | grep -oP 'step_\d+' | grep -oP '\d+')
            echo "[$(date)] Training in progress... step ~$STEP"
        else
            echo "[$(date)] Training in progress... (waiting for first checkpoint)"
        fi
    fi
    sleep 60
done
