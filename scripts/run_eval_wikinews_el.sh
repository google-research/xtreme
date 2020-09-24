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
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='wikinews_el'
MAXL=128

LC=""
MODEL_TYPE="bert-retrieval"
DIM=768

OUT=$OUT_DIR/$TASK/${MODEL}/
mkdir -p $OUT

python $REPO/third_party/evaluate_retrieval.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --embed_size $DIM \
  --batch_size 100 \
  --task_name $TASK \
  $LC \
  --max_seq_length $MAXL \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUT \
