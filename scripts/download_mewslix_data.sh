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

set -eu
REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR

function download_mewslix {
  OUTPATH=$DIR/mewslix
  OUTPATH_TMP=$OUTPATH/tmp
  mkdir -p $OUTPATH_TMP
  pushd $OUTPATH_TMP
  # BEGIN GOOGLE-INTERNAL
  # The relevant subdirectory in the shared 'google-research' repo can be
  # downloaded with the svn command below, but it was not immediately obvious
  # how to install svn to the conda environment/config.
  # svn export https://github.com/google-research/google-research/trunk/dense_representations_for_entity_retrieval
  # cd dense_representations_for_entity_retrieval/mel
  #
  # As workaround, just download the whole zipped repo (182MB as of 6/2022).
  # END GOOGLE-INTERNAL
  wget -O google-research-master.zip \
      https://github.com/google-research/google-research/archive/refs/heads/master.zip
  unzip -q google-research-master.zip
  pushd google-research-master/dense_representations_for_entity_retrieval/mel

  # Run the Mewsli-X downloader script.
  bash get-mewsli-x.sh
  INTERIM_DIR=$PWD/mewsli_x/output/dataset

  echo
  mv $INTERIM_DIR/candidate_set_entities.jsonl $OUTPATH/
  mv $INTERIM_DIR/wikipedia_pairs-{train,dev}.jsonl $OUTPATH/
  mv $INTERIM_DIR/wikinews_mentions-{dev,test}.jsonl $OUTPATH/
  popd
  popd

  # rm -rf $OUTPATH_TMP
  echo "Successfully downloaded data at $OUTPATH" >> $DIR/download.log
}

download_mewslix