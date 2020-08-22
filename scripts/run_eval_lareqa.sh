#!/bin/bash
REPO=$PWD
MODEL="bert-base-multilingual-cased"
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='lareqa'
MAXL=128
TL='en'

NLAYER=12
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert-retrieval"
  DIM=768
fi

OUT=$OUT_DIR/$TASK/${MODEL}/
mkdir -p $OUT

python $REPO/third_party/evaluate_retrieval.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --embed_size $DIM \
  --batch_size 100 \
  --task_name $TASK \
  --num_layers $NLAYER \
  $LC \
  --pool_type cls \
  --max_seq_length $MAXL \
  --data_dir $DATA_DIR \
  --output_dir $OUT \
  --extract_embeds \
  --dist cosine \
  --specific_layer=10