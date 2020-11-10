 #!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD

# To evaluate a checkpoint from fine-tuning (recommended), find the checkpoint
# in the output directory after running scripts/train_lareqa.sh
MODEL="runs/lareqa_mbert_seq352_lr1e-4_b32_fp16_ep3/checkpoint-1000"
# To evaluate a pretrained model *without* finetuning (not recommended):
# MODEL="bert-base-multilingual-cased"  

GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='lareqa'

MODEL_TYPE="bert-retrieval"
DIM=768
DO_LOWER_CASE=""
# These settings should match those used in scripts/train_lareqa.sh
MAX_SEQ_LENGTH=352
MAX_QUERY_LENGTH=96
MAX_ANSWER_LENGTH=256

OUT=$OUT_DIR/$TASK/${MODEL}/
mkdir -p $OUT

python $REPO/third_party/evaluate_retrieval.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --embed_size $DIM \
  --batch_size 100 \
  --task_name $TASK \
  --pool_type cls \
  --max_seq_length $MAX_SEQ_LENGTH \
  --max_query_length $MAX_QUERY_LENGTH \
  --max_answer_length $MAX_ANSWER_LENGTH \
  --data_dir $DATA_DIR \
  --output_dir $OUT \
  --extract_embeds \
  --dist cosine \
  $DO_LOWER_CASE
