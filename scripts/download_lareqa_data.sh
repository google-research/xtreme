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
DIR=$REPO/download
mkdir -p $DIR

function download_lareqa_github {
    OUTPATH=$DIR/lareqa/
    mkdir -p $OUTPATH
    cd $OUTPATH
    wget https://github.com/google-research-datasets/lareqa/archive/master.zip
    unzip master.zip
    mv lareqa-master/* .
    rm -rf lareqa-master/
    rm master.zip
    rm LICENSE
    rm README.md
    echo "Successfully downloaded data at $OUTPATH" >> $DIR/download.log
}

download_lareqa_github
