#!/bin/bash
DATASET="$1"
MODEL="$2"
GPUS="${3:-4}"
MEMORY="$((GPUS * 32))G"
JOB_NAME="Rec-${MODEL}-${DATASET}"
OUTPUT_FILE="${DATASET}-${MODEL}.log"
CPUS="15"
NODE="${4:-}"

echo "Submitting job with the following parameters:"
echo "  Dataset:        $DATASET"
echo "  Model:          $MODEL"
echo "  GPUs:           $GPUS"
echo "  Memory:         $MEMORY"
echo "  Job Name:       $JOB_NAME"
echo "  Output File:    $OUTPUT_FILE"
echo "  CPUs:           $CPUS"
if [ -n "$NODE" ]; then
  echo "  Node:           $NODE"
fi

sbatch_command="sbatch --job-name=\"$JOB_NAME-Training\" \
       --output=\"$OUTPUT_FILE\" \
       --gres=gpu:$GPUS \
       --mem=$MEMORY \
       --cpus-per-task=$CPUS"

if [ -n "$NODE" ]; then
  sbatch_command="$sbatch_command --nodelist=\"$NODE\""
fi

sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif python3 main.py -d \"$DATASET\" -m \"$MODEL\"\""
eval $sbatch_command

sbatch_command_eval="sbatch --job-name=\"$JOB_NAME-Eval\" \
       --output=\"$OUTPUT_FILE-Eval\" \
       --gres=gpu:1 \
       --mem=32 \
       --cpus-per-task=10"

if [ -n "$NODE" ]; then
  sbatch_command_eval="$sbatch_command_eval --nodelist=\"$NODE\""
fi

sbatch_command_eval="$sbatch_command_eval --wrap=\"singularity exec --nv recbole.sif python3 main.py -e True -d \"$DATASET\" -m \"$MODEL\"\""
eval $sbatch_command_eval