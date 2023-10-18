#!/bin/bash
# Run as: bash experiments/memit/baseline_ft/run.sh

MODEL_NAME="meta-llama/Llama-2-7b-hf"
HPARAMS_FN="llama2-7b_wd-1.json"
# MODEL_NAME="EleutherAI/gpt-j-6B"
# HPARAMS_FN="EleutherAI_gpt-j-6B_wd.json"
ALG_NAME="FT"
DATASET="mcf"  # mcf,cf,zsre

export TRANSFORMERS_CACHE="/projects/nlp/data/pmh864/checkpoints/backbones/huggingface/"

MEMIT_DIR="third_party/memit"
# NOTE: MEMIT paths are defined in third_party/memit/globals.yml

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.6
eval "$(conda shell.bash hook)"
conda deactivate
conda activate memit


cd $MEMIT_DIR
python -m experiments.evaluate \
    --alg_name ${ALG_NAME} \
    --model_name ${MODEL_NAME} \
    --hparams_fname ${HPARAMS_FN} \
    --ds_name ${DATASET} \
    --num_edits 10000 \
    --use_cache

conda deactivate
