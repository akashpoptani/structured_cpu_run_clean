#!/usr/bin/env bash
#SBATCH --job-name=TPCHECK
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --output=results_clean/logs/%x_%j.out
#SBATCH --error=results_clean/logs/%x_%j.err

set -euo pipefail

echo "Dry-run generated sbatch. Not a real runner yet."
echo "TAG: TPCHECK"
echo "RUN_MODE: verify"
echo "SHARDING_MODE: tp2"
echo "ACTIVE_MODEL_PATH: /scratch/ramanvr_root/ramanvr/akashpt/models/deepseek-v3.2"
echo "LIN_TOKENS / LOUT_TOKENS / BATCH_SIZE: 10 / 15 / 1"
echo "TODO: load modules"
echo "TODO: set OMP env vars"
echo "TODO: launch torch.distributed.run"
echo "TODO: call clean verification/benchmark runner"
exit 0
