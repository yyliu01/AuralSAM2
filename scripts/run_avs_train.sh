#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_BASE="${REPO_ROOT}/avs.code"
cd "${SCRIPT_DIR}"

DEFAULT_GPUS=4
OMP_THREADS=8

# Reference hyper-parameter table (for quick view)
EPOCH_V1S=140
EPOCH_V1M=140
EPOCH_V2=90

WEIGHT_V1S=3.0
WEIGHT_V1M=3.0
WEIGHT_V2=3.0

print_table() {
  echo "+-------------+------------+------------+------------+"
  echo "| hyper-param |    v1s     |    v1m     |     v2     |"
  echo "+-------------+------------+------------+------------+"
  printf "| %-11s | %-10s | %-10s | %-10s |\n" "epoch" "${EPOCH_V1S}" "${EPOCH_V1M}" "${EPOCH_V2}"
  printf "| %-11s | %-10s | %-10s | %-10s |\n" "weight" "${WEIGHT_V1S}" "${WEIGHT_V1M}" "${WEIGHT_V2}"
  printf "| %-11s | %-10s | %-10s | %-10s |\n" "gpus(def)" "${DEFAULT_GPUS}" "${DEFAULT_GPUS}" "${DEFAULT_GPUS}"
  echo "+-------------+------------+------------+------------+"
}

usage() {
  echo "Usage: $0 <v1s|v1m|v2> [gpus]"
  echo "Example: $0 v1s"
  echo "Example: $0 v2 8"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  print_table
  exit 1
fi

DATASET="$1"
GPUS="${2:-${DEFAULT_GPUS}}"

case "${DATASET}" in
  v1s)
    CODE_DIR="v1s.code"
    EPOCHS="${EPOCH_V1S}"
    ;;
  v1m)
    CODE_DIR="v1m.code"
    EPOCHS="${EPOCH_V1M}"
    ;;
  v2)
    CODE_DIR="v2.code"
    EPOCHS="${EPOCH_V2}"
    ;;
  *)
    echo "Error: dataset must be one of v1s / v1m / v2, got: ${DATASET}"
    echo
    print_table
    exit 1
    ;;
esac

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [[ "${GPUS}" -le 0 ]]; then
  echo "Error: gpus must be a positive integer, got: ${GPUS}"
  exit 1
fi

if [[ ! -f "${CODE_BASE}/${CODE_DIR}/main.py" ]]; then
  echo "Error: training entry not found: ${CODE_BASE}/${CODE_DIR}/main.py"
  exit 1
fi

export OMP_NUM_THREADS="${OMP_THREADS}"

LOG_FILE="train_${DATASET}.log"
CMD=(python3 "${CODE_BASE}/${CODE_DIR}/main.py" --epochs="${EPOCHS}" --gpus="${GPUS}")

echo "Training job is about to start:"
echo "  dataset: ${DATASET}"
echo "  code:    ${CODE_BASE}/${CODE_DIR}/main.py"
echo "  epochs:  ${EPOCHS}"
echo "  gpus:    ${GPUS}"
echo "  log:     ${SCRIPT_DIR}/${LOG_FILE}"
echo
print_table
echo
echo "Command: nohup ${CMD[*]} > ${LOG_FILE} 2>&1 &"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo "Training started in background, PID: $!"
