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

# Fine-tune a pretrained multilingual encoder on the LAReQA QA retrieval task.

# If using an uncased model, add:
# --do_lower_case

# Mixed precision training (--fp16) requires Apex. To install, see:
# https://anaconda.org/conda-forge/nvidia-apex

python third_party/run_retrieval_qa.py \
  --model_type bert-retrieval \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --train_file download/squad/train-v1.1.json \
  --predict_file download/squad/dev-v1.1.json \
  --per_gpu_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --max_seq_length 352 \
  --max_query_length 96 \
  --max_answer_length 256 \
  --save_steps 100 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --warmup_steps 10 \
  --fp16 \
  --output_dir runs/lareqa_mbert_seq352_lr1e-4_b32_fp16_ep3 \
  --weight_decay 0.0 \
  --threads 8 \
  --train_lang en \
  --eval_lang en
