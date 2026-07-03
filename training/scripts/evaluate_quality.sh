#!/bin/bash
# Run full evaluation on the quality model once training completes.
# Usage: bash training/scripts/evaluate_quality.sh [collect_steps] [epochs]

set -e
cd "$(dirname "$0")/../.."

MODEL="checkpoints/tinystories_quality.blmf"
TOKENIZER="checkpoints/tinystories_vocab4096.json"
TRAIN_DATA="data/tinystories_train.txt"
VAL_DATA="data/tinystories_val.txt"
TREE_OUTPUT="/tmp/ste_quality_tree.bin"
RESULTS_JSON="/tmp/quality_results.json"
COLLECT_STEPS="${1:-100000}"
EPOCHS="${2:-500}"

echo "=============================================="
echo "Quality Model Evaluation Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Steps: $COLLECT_STEPS, Epochs: $EPOCHS"
echo ""

# Step 1: Verify model exists
if [ ! -f "$MODEL" ]; then
    echo "[!] Model not found at $MODEL"
    echo "    Quality training may still be running."
    echo "    Check: screen -r quality_train"
    exit 1
fi

# Step 2: Train STE tree head
echo "[1/3] Training STE tree head..."
python3 -u training/scripts/train_tree_on_model.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --data "$TRAIN_DATA" \
    --output "$TREE_OUTPUT" \
    --collect-steps "$COLLECT_STEPS" \
    --epochs "$EPOCHS" \
    --lr 0.01 \
    --weighting sqrt_inv 2>&1 | tee /tmp/ste_quality_training.log

echo ""
echo "[2/3] Running comprehensive evaluation..."
python3 -u training/scripts/eval_tree_head.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --data "$VAL_DATA" \
    --max-stories 500 \
    --output "$RESULTS_JSON" 2>&1 | tee /tmp/quality_eval.log

echo ""
echo "[3/3] Generating samples..."
python3 -u training/scripts/generate_samples.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "Once upon a time" \
    --length 50 \
    --output /tmp/quality_samples.txt 2>&1

echo ""
echo "=============================================="
echo "Evaluation Complete"
echo "Results: $RESULTS_JSON"
echo "Tree: $TREE_OUTPUT"
echo "Samples: /tmp/quality_samples.txt"
echo "=============================================="
