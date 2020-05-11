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
DIR=$REPO/download/
mkdir -p $DIR

# download XNLI dataset
function download_xnli {
    OUTPATH=$DIR/xnli-tmp/
    if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH
      fi
      unzip $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [ ! -d $OUTPATH/XNLI-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH
      fi
      unzip $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi
    python $REPO/utils_preprocess.py \
      --data_dir $OUTPATH \
      --output_dir $DIR/xnli/ \
      --task xnli
    rm -rf $OUTPATH 
}

# download PAWS-X dataset
function download_pawsx {
    cd $DIR
    wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz
    tar xzvf x-final.tar.gz -C $DIR/
    python $REPO/utils_preprocess.py \
      --data_dir $DIR/x-final \
      --output_dir $DIR/pawsx/ \
      --task pawsx
    rm -rf x-final x-final.tar.gz
}

# download UD-POS dataset
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll/
    mkdir -p $out_dir
    cd $base_dir
    curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
    tar -xzvf $base_dir/ud-treebanks-v2.5.tgz
    
    langs=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh)
    for x in $base_dir/ud-treebanks-v2.5/*/*.conllu; do
        file="$(basename $x)"
        IFS='_' read -r -a array <<< "$file"
        lang=${array[0]}
        if [[ " ${langs[@]} " =~ " ${lang} " ]]; then
            lang_dir=$out_dir/$lang/
            mkdir -p $lang_dir
            y=$lang_dir/${file/conllu/conll}
            if [ ! -f "$y" ]; then
                echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
                python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms
            else
                echo "${y} exists"
            fi
        else
            echo "skip language $lang"
        fi
    done

    python $REPO/utils_preprocess.py --data_dir $out_dir/ --output_dir $DIR/udpos/ --task  udpos 
    rm -rf $out_dir ud-treebanks-v2.tgz $DIR/udpos-tmp
}

function download_panx {
    echo "Download panx NER dataset"
    if [ -f $DIR/AmazonPhotos.zip ]; then
        unzip $DIR/AmazonPhotos.zip -d $DIR/
        base_dir=$DIR/panx_dataset/ && cd $base_dir
        langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu)
        for lg in ${langs[@]}; do
            tar xzf $base_dir/${lg}.tar.gz
            for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
        done
        python $REPO/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DIR/panx \
            --task panx
        rm -rf $base_dir
    else
        echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive mannually and save it to $DIR/AmazonPhotos.zip"
        echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
    fi
}

function download_tatoeba {
    base_dir=$DIR/tatoeba-tmp/
    wget https://github.com/facebookresearch/LASER/archive/master.zip
    unzip -o master.zip -d $base_dir/
    mv $base_dir/LASER-master/data/tatoeba/v1/* $base_dir/
    python $REPO/utils_preprocess.py \
      --data_dir $base_dir \
      --output_dir $DIR/tatoeba \
      --task tatoeba
    rm -rf $base_dir master.zip
}

function download_bucc18 {
    base_dir=$DIR/bucc2018/
    cd $DIR
    for lg in zh ru de fr; do
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.training-gold.tar.bz2
        tar xjf bucc2018-${lg}-en.training-gold.tar.bz2
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.sample-gold.tar.bz2
        tar xjf bucc2018-${lg}-en.sample-gold.tar.bz2
    done
    mv $base_dir/*/* $base_dir/
    for f in $base_dir/*training*; do mv $f ${f/training/test}; done
    for f in $base_dir/*sample*; do mv $f ${f/sample/dev}; done
    rm -rf $base_dir/*test.gold $DIR/bucc2018*tar.bz2 $base_dir/{zh,ru,de,fr}-en/
}


function download_squad {
    echo "download squad"
    base_dir=$DIR/squad/
    mkdir -p $base_dir && cd $base_dir
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
    # Download the SQuAD evaluation script (used for XQuAD and TyDiQA-GoldP)
    wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
}

function download_xquad {
    echo "download xquad"
    base_dir=$DIR/xquad/
    mkdir -p $base_dir && cd $base_dir
    for lang in ar de el en es hi ru th tr vi zh; do
      wget https://raw.githubusercontent.com/deepmind/xquad/master/xquad.${lang}.json
    done
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $base_dir --task xquad
}

function download_mlqa {
    echo "download mlqa"
    base_dir=$DIR/mlqa/
    mkdir -p $base_dir && cd $base_dir
    zip_file=MLQA_V1.zip
    wget https://dl.fbaipublicfiles.com/MLQA/${zip_file}
    unzip ${zip_file}
    rm ${zip_file}
    # Download the MLQA evaluation script
    wget https://raw.githubusercontent.com/facebookresearch/MLQA/master/mlqa_evaluation_v1.py
    python $REPO/utils_preprocess.py --data_dir $base_dir/MLQA_V1/test --output_dir $base_dir --task mlqa
}

function download_tydiqa {
    echo "download tydiqa-goldp"
    base_dir=$DIR/tydiqa/
    mkdir -p $base_dir && cd $base_dir
    tydiqa_train_file=tydiqa-goldp-v1.1-train.json
    tydiqa_dev_file=tydiqa-goldp-v1.1-dev.tgz
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_train_file}
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_dev_file}
    tar -xvf ${tydiqa_dev_file}
    rm ${tydiqa_dev_file}
    out_dir=$base_dir/tydiqa-goldp-v1.1-train
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $out_dir --task tydiqa
    mv $base_dir/$tydiqa_train_file $out_dir/
}

download_xnli
download_pawsx
download_udpos
download_panx
download_tatoeba
download_bucc18
download_squad
download_xquad
download_mlqa
download_tydiqa
