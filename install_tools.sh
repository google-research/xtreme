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
LIB=$REPO/third_party
mkdir -p $LIB

# install conda env
conda create --name xtreme --file conda-env.txt
conda init bash
source activate xtreme

# install latest transformer
cd $LIB
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
cd $LIB

pip install seqeval
pip install tensorboardx

# install XLM tokenizer
pip install sacremoses
pip install pythainlp
pip install jieba

git clone https://github.com/neubig/kytea.git && cd kytea
autoreconf -i
./configure --prefix=$HOME/local
make && make install
pip install kytea
cd $LIB

wget -O $LIB/evaluate_squad.py https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
wget -O $LIB/evaluate_mlqa.py https://raw.githubusercontent.com/facebookresearch/MLQA/master/mlqa_evaluation_v1.py
