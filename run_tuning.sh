#!/bin/bash

# Activate virtual environment
source /scratch/vmm5481/CV/pose_env/bin/activate

# Update these paths to match your lab server
export DATA_ROOT='/scratch/vmm5481/CV/dataset'
export VPOSER_DIR="/scratch/vmm5481/CV/VPoserModelFiles/vposer_v2_05"
export OUTPUT_DIR="./tuning_results"

# Edit the script to update paths
sed -i "s|DATA_ROOT = '/path/to/dataset'|DATA_ROOT = '$DATA_ROOT'|g" hyperparameter_tuning.py
sed -i "s|VPOSER_DIR = '/path/to/vposer_v2_05'|VPOSER_DIR = '$VPOSER_DIR'|g" hyperparameter_tuning.py
sed -i "s|OUTPUT_DIR = './tuning_results'|OUTPUT_DIR = '$OUTPUT_DIR'|g" hyperparameter_tuning.py

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the tuning script
python3 hyperparameter_tuning.py

# Deactivate virtual environment when done
deactivate
