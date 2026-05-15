#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${REPO_ROOT}/ref-avs.code"
cd "${SCRIPT_DIR}"

DEFAULT_GPUS=4
DEFAULT_EPOCHS=50
DEFAULT_LR=1e-4
OMP_THREADS=8

print_table() {
  echo "+-------------+----------------+"
  echo "| hyper-param |   ref-avs      |"
  echo "+-------------+----------------+"
  printf "| %-11s | %-14s |\n" "epoch" "${DEFAULT_EPOCHS}"
  printf "| %-11s | %-14s |\n" "lr" "${DEFAULT_LR}"
  printf "| %-11s | %-14s |\n" "gpus(def)" "${DEFAULT_GPUS}"
  echo "+-------------+----------------+"
}

usage() {
  echo "Usage: $0 [gpus]"
  echo "Example: $0"
  echo "Example: $0 8"
}

if [[ $# -gt 1 ]]; then
  usage
  print_table
  exit 1
fi

GPUS="${1:-${DEFAULT_GPUS}}"

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [[ "${GPUS}" -le 0 ]]; then
  echo "Error: gpus must be a positive integer, got: ${GPUS}"
  exit 1
fi

if [[ ! -f "${CODE_DIR}/main.py" ]]; then
  echo "Error: training entry not found: ${CODE_DIR}/main.py"
  exit 1
fi

export OMP_NUM_THREADS="${OMP_THREADS}"

LOG_FILE="train_ref_avs.log"
CMD=(
  python3 "${CODE_DIR}/main.py"
  --epochs="${DEFAULT_EPOCHS}"
  --gpus="${GPUS}"
  --lr="${DEFAULT_LR}"
)

echo "Training job is about to start:"
echo "  dataset: ref-avs (REFAVS)"
echo "  code:    ${CODE_DIR}/main.py"
echo "  epochs:  ${DEFAULT_EPOCHS}"
echo "  lr:      ${DEFAULT_LR}"
echo "  gpus:    ${GPUS}"
echo "  log:     ${SCRIPT_DIR}/${LOG_FILE}"
echo
print_table
echo
echo "Command: nohup ${CMD[*]} > ${LOG_FILE} 2>&1 &"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo "Training started in background, PID: $!"
