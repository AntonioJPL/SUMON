#!/bin/bash

CONDA_INIT_SCRIPT="/opt/lst-drive/miniconda3/conda_init.sh"

# Check if Conda init script exists
if [ ! -f "$CONDA_INIT_SCRIPT" ]; then
    echo "Error: Conda init script not found at $CONDA_INIT_SCRIPT"
    exit 1
fi

# Initialize Conda
source "$CONDA_INIT_SCRIPT"

# Path to environment.yml file
ENV_FILE="/opt/lst-drive/src/SUMON/environment.yml"
# Check if environment.yml file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: environment file not found"
    exit 1
fi

# Create Conda environment using the environment.yml file
conda env create -f "$ENV_FILE"

# Display a message indicating successful environment creation
echo "Conda environment created successfully!"

ENV_NAME="moveEnv"

if conda info --envs | grep -q "^$ENV_NAME[[:space:]]"; then
    conda activate $ENV_NAME
else
    conda create -y -n "$ENV_NAME"
    conda activate $ENV_NAME
fi

#conda install -y pip

#pip uninstall ipinfo
#pip install python-dotenv
#pip install -r /opt/lst-drive/src/LST-DM-LP-Internal/requirements.txt
