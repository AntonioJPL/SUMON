#!/usr/bin/env bash

# === CONFIGURATION ===
if [ -n "${BASH_SOURCE-}" ]; then
  SCRIPT="$BASH_SOURCE"
else
  SCRIPT="$0"
fi
PROJECT_DIR="$(cd "$(dirname "$SCRIPT")" && pwd -P)"
CONDA_ENV="moveEnv"
LOG_FILE="$PROJECT_DIR/cron_plots.log"

# === ACTIVATE CONDA ENVIRONMENT ===
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# === CHANGE TO PROJECT DIRECTORY ===
cd "$PROJECT_DIR"

# === GET CURRENT DATE ===
DAY=$(date +%d)
MONTH=$(date +%m)
STAMP="$(date -Is)"

echo "$STAMP :: Checking date $(date +%F)" >> "$LOG_FILE"

# === RUN MONTHLY SCRIPT ON THE 1st OF EVERY MONTH ===
if [ "$DAY" -eq 13 ]; then
    echo "$STAMP :: Running monthly script..." >> "$LOG_FILE"
    python monthlyMovementZH.py
    # === RUN ANNUAL SCRIPT ONLY ON JANUARY 1st ===
    if [ "$MONTH" -eq 1 ]; then
        echo "$STAMP :: Running annual script..." >> "$LOG_FILE"
        python yearlyMovementZH.py
    fi
fi
