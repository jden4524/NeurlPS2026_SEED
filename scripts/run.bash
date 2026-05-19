#!/bin/bash

# ==========================================
# ⚙️ SEED-V2 multimodal active-injection evaluation launcher script
# Please run from the project root: bash scripts/run.bash
# Or you can also run it directly from inside the scripts directory!
# ==========================================

# ==========================================
# 📍 0. Smart absolute-path resolution (Bash foolproof design)
# ==========================================
# Get the absolute path of the scripts directory that contains run.bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go up one level to get the SEED project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Lock in the absolute path of run.py
RUN_PY_PATH="$PROJECT_ROOT/run.py"

# cd into the project root before execution so relative paths (e.g., outputs/) are generated correctly
cd "$PROJECT_ROOT" || exit 1

# 1. Hardware resource allocation
export CUDA_VISIBLE_DEVICES="0,1" # Adjust as needed, e.g., "0,1,2,3"

# 2. Core paths and mode
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
DATASET_MODE="local"     # "local" or "hf"

# 3. 🌟 SEED-V2 full hyperparameters 🌟
WINDOW_START=5           # Trigger window start (protects very early generation)
WINDOW_END=60            # Trigger window end (protects later convergence stage)
ENTROPY_THRESHOLD=0.25   # Absolute entropy threshold that triggers injection
INJECTION_ALPHA=0.5      # Injection strength of the visual feature e_vis
INJECTION_STEPS=1        # Number of sustained injection steps after triggering (K)

MAX_TOKENS=512           # Maximum number of generated tokens

# 4. Experiment control switches
USE_SEED="true"      # Set to "true" to enable active injection, "false" to run Baseline
RESUME="false"        # Set to "true" to enable checkpoint resume, "false" to start from scratch

# Assemble Python run arguments automatically based on the switches
EXTRA_FLAGS=""

echo "==========================================="
if [ "$USE_SEED" = "false" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --disable_seed"
    echo "⚠️  SEED intervention disabled, running the Baseline model"
else
    echo "✅ SEED active-injection intervention enabled"
fi

if [ "$RESUME" = "false" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --disable_resume"
    echo "⚠️  Checkpoint resume disabled, evaluation will start from scratch"
else
    echo "✅ Checkpoint resume enabled"
fi
echo "==========================================="

echo "🚀 Launching SEED-V2 evaluation pipeline"
echo "📂 Project root: $PROJECT_ROOT"
echo "🖥️  Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "📦 Model path: $MODEL_PATH"
echo "🎯 Trigger window: steps [$WINDOW_START, $WINDOW_END]"
echo "📊 Trigger entropy threshold: $ENTROPY_THRESHOLD"
echo "💉 Injection strength Alpha: $INJECTION_ALPHA (sustained $INJECTION_STEPS steps)"
echo "==========================================="

# 5. Invoke the Python main program (using absolute path)
python "$RUN_PY_PATH" \
    --model_path "$MODEL_PATH" \
    --dataset_mode "$DATASET_MODE" \
    --window_start $WINDOW_START \
    --window_end $WINDOW_END \
    --entropy_threshold $ENTROPY_THRESHOLD \
    --injection_alpha $INJECTION_ALPHA \
    --injection_steps $INJECTION_STEPS \
    --max_new_tokens $MAX_TOKENS \
    $EXTRA_FLAGS