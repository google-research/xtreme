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