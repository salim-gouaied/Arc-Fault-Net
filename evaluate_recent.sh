#!/bin/bash

echo "Starting evaluation of models trained since 22/06..."

# Loop over run directories starting from 20260622
for RUN_DIR in runs/*_2026062[2-9]_* runs/*_2026063[0-1]_* runs/*_202607*; do
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
echo "Evaluation of all recent models completed."
