#!/bin/bash

GPUS=1
OVERSAMPLE="0.0"
UNDERSAMPLE="0.0"

while getopts ":c:d:m:erzg:s:n:o:u:h" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    e) EVAL="-e" ;;
    r) RERANK="-r" ;;
    z) ZIPF="-z" ;;
    g) GPUS="$OPTARG" ;;
    s) NAME="$OPTARG" ;;
    n) NODE="$OPTARG" ;;
    o) OVERSAMPLE="$OPTARG" ;;
    u) UNDERSAMPLE="$OPTARG" ;;
    h)
      echo "Usage: $0 -d <DATASET> -m <MODEL> [-g <GPUS=1>] [-n <NODE>] [-o <OVERSAMPLE=0.0>] [-u <UNDERSAMPLE=0.0>] [-e] [-r] [-z]"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

MEMORY="$((GPUS * 24))G"
if [ -n "$DATASET" ] && [ -n "$MODEL" ]; then
  JOB_NAME="Rec-${MODEL}-${DATASET}"
  OUTPUT_FILE="${DATASET}-${MODEL}"
else
  JOB_NAME="Rec-Multi${CONFIG_FILE}"
  OUTPUT_FILE="Multi${CONFIG_FILE}"
fi

CPUS="$((GPUS * 10))"
if (( $(echo "$OVERSAMPLE != 0.0" | bc) )); then
  JOB_NAME="${JOB_NAME}_O_${OVERSAMPLE}"
  OUTPUT_FILE="${OUTPUT_FILE}_O_${OVERSAMPLE}"
fi
if (( $(echo "$UNDERSAMPLE != 0.0" | bc) )); then
  JOB_NAME="${JOB_NAME}_U_${UNDERSAMPLE}"
  OUTPUT_FILE="${OUTPUT_FILE}_U_${UNDERSAMPLE}"
fi

if [ -n "$NAME" ]; then
  JOB_NAME="${NAME}"
fi

if [ -n "$ZIPF" ]; then
  JOB_NAME="${JOB_NAME}_Zipf"
  OUTPUT_FILE="${OUTPUT_FILE}_Zipf"
fi

if [ -n "$RERANK" ]; then
  JOB_NAME="${JOB_NAME}_MMR"
  OUTPUT_FILE="${OUTPUT_FILE}_MMR"
fi

if [ -n "$EVAL" ]; then
  JOB_NAME="${JOB_NAME}_Eval"
  OUTPUT_FILE="${OUTPUT_FILE}_Eval"
fi
OUTPUT_FILE="${OUTPUT_FILE}.log"

echo "Submitting job with the following parameters:"
[ -n "$DATASET" ]     && echo "  Dataset:        $DATASET"
[ -n "$MODEL" ]       && echo "  Model:          $MODEL"
[ -n "$CONFIG_FILE" ] && echo "  Config File:    $CONFIG_FILE"
[ -n "$NAME" ] && echo "  Name:           $NAME"
echo "  GPUs:           $GPUS"
echo "  Memory:         $MEMORY"
echo "  Job Name:       $JOB_NAME"
echo "  Output File:    $OUTPUT_FILE"
echo "  CPUs:           $CPUS"
[ -n "$NODE" ] && echo "  Node:           $NODE"
(( $(echo "$OVERSAMPLE != 0.0" | bc) ))  && echo "  Oversample:     $OVERSAMPLE"
(( $(echo "$UNDERSAMPLE != 0.0" | bc) )) && echo "  Undersample:    $UNDERSAMPLE"
[ -n "$EVAL" ] && echo "  Evaluation:     Enabled"
[ -n "$RERANK" ] && echo "  MMR Rerank:     Enabled"
[ -n "$ZIPF" ] && echo "  ZIPS Penalty:   Enabled"

sbatch_command="sbatch --job-name=\"$JOB_NAME\" --output=\"$OUTPUT_FILE\" --gres=gpu:$GPUS --mem=$MEMORY --cpus-per-task=$CPUS"

if [ -z "${NAME+x}" ] && [ -n "${JOB_NAME+x}" ]; then
  NAME="${JOB_NAME}"
fi

[ -n "$NODE" ] && sbatch_command+=" --nodelist=\"$NODE\""
if [ -n "$DATASET" ] && [ -n "$MODEL" ]; then
  sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif bash run_model.sh -d $DATASET -m $MODEL -s $NAME -o $OVERSAMPLE -u $UNDERSAMPLE $EVAL $RERANK $ZIPF\""
else
  sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif bash run_model.sh -c $CONFIG_FILE\""
fi
eval "$sbatch_command"
