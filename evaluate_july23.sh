#!/bin/bash

echo "Starting evaluation of models trained on July 23rd..."

# Loop over run directories from July 23, 2026
for RUN_DIR in runs/*_20260724_*; do
    # Check if it's a valid directory with a best model checkpoint
    if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/best_single.pt" ]; then
        echo ""
        echo "========================================================="
        echo " Evaluating: $RUN_DIR"
        echo "========================================================="
        venv/bin/python mini_evaluate.py --run "$RUN_DIR"
    fi
done

echo ""
echo "Evaluation of July 23 models completed."
