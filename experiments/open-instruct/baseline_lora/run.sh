#!/bin/bash
# Run as: bash experiments/open-instruct/baseline_lora/run.sh

MODEL_HF="meta-llama/Llama-2-7b-hf"
MODEL_NAME="llama-2-7b"

BASE_DIR="/projects/nlp/data/data/fact_mutability/third_party/open-instruct/"
OUTS_DIR="${BASE_DIR}/outputs"
DATA_DIR="${BASE_DIR}/data/processed/tulu_v2/oasst1_subset"
DSET_FN="oasst1"

NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

export TRANSFORMERS_CACHE="/projects/nlp/data/pmh864/checkpoints/backbones/huggingface/"

OI_DIR="third_party/open-instruct"

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.6
eval "$(conda shell.bash hook)"
conda deactivate
conda activate open-instruct

echo "Training ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


cd $OI_DIR

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${MODEL_HF} \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name ${MODEL_HF} \
    --use_slow_tokenizer \
    --train_file ${DATA_DIR}/${DSET_FN}_data.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir ${OUTS_DIR}/${DSET_FN}_${MODEL_NAME}_lora/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path ${MODEL_HF} \
    --lora_model_name_or_path ${OUTS_DIR}/${DSET_FN}_${MODEL_NAME}_lora/ \
    --output_dir ${OUTS_DIR}/${DSET_FN}_${MODEL_NAME}_lora_merged/ \
    --save_tokenizer
