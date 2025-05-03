#!/bin/bash

## usage
# bash scripts/run.sh [--force-reinstall] > log.txt 2>&1

## usage sandbox
# bash scripts/run.sh --force-reinstall 2>&1 | tee log.txt
# bash scripts/run.sh 2>&1 | tee log.txt

## Exit immediately if a command exits with a non-zero status.
set -e

## setup
source ~/.bashrc

## parse arguments
FORCE_REINSTALL=false
for arg in "$@"; do
    if [[ "$arg" == "--force-reinstall" ]]; then
        FORCE_REINSTALL=true
    fi
done

## assertions
os_name="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$os_name" != "linux" ]]; then
    echo "Error: Expected Linux, but got $os_name" >&2
    exit 1
fi

## info
echo "Base directory: $(mamba info --base)"
echo "Current shell: $SHELL"
echo "Available environments:"
conda env list

## set up the conda environment
. "/home/pschaefer/miniforge3/etc/profile.d/conda.sh"
. "/home/pschaefer/miniforge3/etc/profile.d/mamba.sh"
mamba activate base

## remove and recreate environment only if --force-reinstall is set
if [[ "$FORCE_REINSTALL" == true ]]; then
    if conda env list | grep -qE "^partipy\s"; then
        echo "Removing existing 'partipy' environment..."
        mamba env remove -y --name partipy
    else
        echo "'partipy' environment does not exist. No need to remove."
    fi
    echo "Creating 'partipy' environment..."
    mamba env create -y --name partipy --file=conda_envs/default_env.yaml
else
    echo "Skipping environment reinstallation. Use --force-reinstall to recreate the environment."
fi

mamba activate partipy

## run python scripts
echo "python ms_bench.py"
python ms_bench.py

echo "python ms_xenium_bench.py"
python ms_xenium_bench.py
