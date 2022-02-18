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
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [ ! -d $OUTPATH/XNLI-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi
    python $REPO/utils_preprocess.py \
      --data_dir $OUTPATH \
      --output_dir $DIR/xnli/ \
      --task xnli
    rm -rf $OUTPATH
    echo "Successfully downloaded data at $DIR/xnli" >> $DIR/download.log
}

# download PAWS-X dataset
function download_pawsx {
    cd $DIR
    wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz -q --show-progress
    tar xzf x-final.tar.gz -C $DIR/
    python $REPO/utils_preprocess.py \
      --data_dir $DIR/x-final \
      --output_dir $DIR/pawsx/ \
      --task pawsx
    rm -rf x-final x-final.tar.gz
    echo "Successfully downloaded data at $DIR/pawsx" >> $DIR/download.log
}


# Helper function to download the UD-POS data.
# In order to ensure backwards compatibility with the XTREME evaluation,
# languages in XTREME use the UD version used in the original paper; for the new
# languages in XTREME-R, we use a more recent UD version.
function download_treebank {
    base_dir=$2
    out_dir=$3
    if [ $1 == "xtreme" ]; then
      url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
      langs=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh)
      ud_version="2.5"
    elif [ $1 == "xtreme-r" ]; then
      url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
      langs=(lt pl uk wo ro)
      ud_version="2.7"
    else
      echo "$1 is not an accepted argument for downloading the treebank. Accepted values: xtreme, xtreme-r"
      exit 0
    fi
    echo $1
    echo "$url"
    curl -s --remote-name-all "$url"

    tar -xzf $base_dir/ud-treebanks-v$ud_version.tgz

    for x in $base_dir/ud-treebanks-v$ud_version/*/*.conllu; do
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
        fi
    done
}

# Download UD-POS dataset.
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll/
    mkdir -p $out_dir
    cd $base_dir

    download_treebank xtreme $base_dir $out_dir
    download_treebank xtreme-r $base_dir $out_dir

    python $REPO/utils_preprocess.py --data_dir $out_dir/ --output_dir $DIR/udpos/ --task  udpos
    rm -rf $out_dir ud-treebanks-v2.tgz $DIR/udpos-tmp
    echo "Successfully downloaded data at $DIR/udpos" >> $DIR/download.log
}

function download_panx {
    echo "Download panx NER dataset"
    if [ -f $DIR/AmazonPhotos.zip ]; then
        base_dir=$DIR/panx_dataset/
        unzip -qq -j $DIR/AmazonPhotos.zip -d $base_dir
        cd $base_dir
        langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu qu pl uk az lt pa gu ro)
        for lg in ${langs[@]}; do
            tar xzf $base_dir/${lg}.tar.gz
            for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
        done
        cd ..
        python $REPO/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DIR/panx \
            --task panx
        rm -rf $base_dir
        echo "Successfully downloaded data at $DIR/panx" >> $DIR/download.log
    else
        echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive manually and save it to $DIR/AmazonPhotos.zip"
        echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
    fi
}

function download_tatoeba {
    base_dir=$DIR/tatoeba-tmp/
    wget https://github.com/facebookresearch/LASER/archive/main.zip
    unzip -qq -o main.zip -d $base_dir/
    mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
    python $REPO/utils_preprocess.py \
      --data_dir $base_dir \
      --output_dir $DIR/tatoeba \
      --task tatoeba
    rm -rf $base_dir main.zip
    echo "Successfully downloaded data at $DIR/tatoeba" >> $DIR/download.log
}

function download_bucc18 {
    base_dir=$DIR/bucc2018/
    cd $DIR
    for lg in zh ru de fr; do
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.training-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.training-gold.tar.bz2
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.sample-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.sample-gold.tar.bz2
    done
    mv $base_dir/*/* $base_dir/
    for f in $base_dir/*training*; do mv $f ${f/training/test}; done
    for f in $base_dir/*sample*; do mv $f ${f/sample/dev}; done
    rm -rf $base_dir/*test.gold $DIR/bucc2018*tar.bz2 $base_dir/{zh,ru,de,fr}-en/
    echo "Successfully downloaded data at $DIR/bucc2018" >> $DIR/download.log
}

function download_squad {
    echo "download squad"
    base_dir=$DIR/squad/
    mkdir -p $base_dir && cd $base_dir
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -q --show-progress
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json -q --show-progress
    echo "Successfully downloaded data at $DIR/squad"  >> $DIR/download.log
}

function download_xquad {
    echo "download xquad"
    base_dir=$DIR/xquad/
    mkdir -p $base_dir && cd $base_dir
    for lang in ar de el en es hi ru th tr vi zh; do
      wget https://raw.githubusercontent.com/deepmind/xquad/master/xquad.${lang}.json -q --show-progress
    done
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $base_dir --task xquad
    echo "Successfully downloaded data at $DIR/xquad" >> $DIR/download.log
}

function download_mlqa {
    echo "download mlqa"
    base_dir=$DIR/mlqa/
    mkdir -p $base_dir && cd $base_dir
    zip_file=MLQA_V1.zip
    wget https://dl.fbaipublicfiles.com/MLQA/${zip_file} -q --show-progress
    unzip -qq ${zip_file}
    rm ${zip_file}
    python $REPO/utils_preprocess.py --data_dir $base_dir/MLQA_V1/test --output_dir $base_dir --task mlqa
    echo "Successfully downloaded data at $DIR/mlqa" >> $DIR/download.log
}

function download_tydiqa {
    echo "download tydiqa-goldp"
    base_dir=$DIR/tydiqa/
    mkdir -p $base_dir && cd $base_dir
    tydiqa_train_file=tydiqa-goldp-v1.1-train.json
    tydiqa_dev_file=tydiqa-goldp-v1.1-dev.tgz
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_train_file} -q --show-progress
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_dev_file} -q --show-progress
    tar -xf ${tydiqa_dev_file}
    rm ${tydiqa_dev_file}
    out_dir=$base_dir/tydiqa-goldp-v1.1-train
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $out_dir --task tydiqa
    mv $base_dir/$tydiqa_train_file $out_dir/
    echo "Successfully downloaded data at $DIR/tydiqa" >> $DIR/download.log
}

function download_xcopa {
    echo "download xcopa"
    langs=(et ht id it qu sw ta th tr vi zh)
    base_dir=$DIR/xcopa/
    mkdir -p $base_dir && cd $base_dir
    wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip
    unzip COPA.zip
    for split in train val test; do
      mv COPA/${split}.jsonl ${split}-en.jsonl
    done
    rm -r -f COPA.zip COPA
    for lang in ${langs[@]}; do
      wget https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/${lang}/val.${lang}.jsonl -q --show-progress
      wget https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/${lang}/test.${lang}.jsonl -q --show-progress
      mv $base_dir/val.${lang}.jsonl $base_dir/val-${lang}.jsonl
      mv $base_dir/test.${lang}.jsonl $base_dir/test-${lang}.jsonl
    done
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $base_dir --task xcopa
    echo "Successfully downloaded data at $DIR/xcopa" >> $DIR/download.log
}

function download_siqa {
    echo "download siqa"
    base_dir=$DIR/siqa/
    mkdir -p $base_dir && cd $base_dir
    siqa_file=socialIQa_v1.4.tgz
    wget https://maartensap.github.io/social-iqa/data/${siqa_file}
    tar -xf ${siqa_file}
    rm ${siqa_file}
    echo "Successfully downloaded data at $DIR/siqa" >> $DIR/download.log
}

download_xnli
download_pawsx
download_tatoeba
download_bucc18
download_squad
download_xquad
download_mlqa
download_tydiqa
download_udpos
download_panx
download_xcopa
download_siqa
