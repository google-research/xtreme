# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Data processing for XCOPA.

Based on https://github.com/huggingface/datasets/blob/master/datasets/xcopa/xcopa.py
"""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets


_CITATION = """\
  @article{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  journal={arXiv preprint},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}

@inproceedings{roemmele2011choice,
  title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
  booktitle={2011 AAAI Spring Symposium Series},
  year={2011},
  url={https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF},
}
"""

_DESCRIPTION = """\
  XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across
languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around
the globe. The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages. All the details about the
creation of XCOPA and the implementation of the baselines are available in the paper.\n
"""

# Added en language to support training
_LANG = ["en", "et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]

_URL = "https://github.com/cambridgeltl/xcopa/archive/master.zip"

DATA_DIR = '../../download'


class XcopaConfig(datasets.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, **kwargs):
        """

        Args:
            data_dir: directory for the given language dataset
            **kwargs: keyword arguments forwarded to super.
        """
        super(XcopaConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Xcopa(datasets.GeneratorBasedBuilder):
    """TODO(xcopa): Short description of my dataset."""

    # TODO(xcopa): Set up version.
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        XcopaConfig(
            name=lang,
            description="Xcopa language {}".format(lang),
        )
        for lang in _LANG + [f'translate_train_{l}' for l in _LANG if _LANG != "en"] + ['translate_train_all']
    ]

    def _info(self):
        # TODO(xcopa): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION + self.config.description,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "premise": datasets.Value("string"),
                    "choice1": datasets.Value("string"),
                    "choice2": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "idx": datasets.Value("int32"),
                    "changed": datasets.Value("bool"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/cambridgeltl/xcopa",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(xcopa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
#         dl_dir = dl_manager.download_and_extract(_URL)

#         data_dir = os.path.join(dl_dir, "xcopa-master", "data", self.config.name)
        if self.config.name is not None and self.config.name.startswith('translate_train_all'):
            data_dir = os.path.join(DATA_DIR, 'xcopa_translate_train')
            gold_dir = os.path.join(DATA_DIR, 'xcopa')

            return [
              datasets.SplitGenerator(
                  name=datasets.Split.VALIDATION,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(gold_dir, f"val-en.jsonl")},
              ),
              datasets.SplitGenerator(
                  name=datasets.Split.TRAIN,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": [os.path.join(data_dir, f"xcopa_{lang}_translate_train.jsonl") for lang in _LANG if lang != "en"] + [os.path.join(gold_dir, f"train.en.jsonl")]},
              ),
              ]

        if self.config.name.startswith('translate_train'):
            lang = self.config.name.split('_')[-1]
            data_dir = os.path.join(DATA_DIR, 'xcopa_translate_train')
            gold_dir = os.path.join(DATA_DIR, 'xcopa')

            splits =  [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": os.path.join(data_dir, f"xcopa_{lang}_translate_val.jsonl")},
                ),

                datasets.SplitGenerator(
                  name=datasets.Split.TRAIN,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, f"xcopa_{lang}_translate_train.jsonl")},
                    ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": os.path.join(gold_dir, "test-" + lang + ".jsonl")},
                      )
            ]
            return splits
        else:
          data_dir = os.path.join(DATA_DIR, 'xcopa')
          splits =  [
              datasets.SplitGenerator(
                  name=datasets.Split.VALIDATION,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "val-" + self.config.name + ".jsonl")},
              ),
          ]
          if self.config.name == 'en':
            splits.append(
                datasets.SplitGenerator(
                  name=datasets.Split.TRAIN,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "train-" + self.config.name + ".jsonl")},
                    )
                )
          else:
            splits.append(
              datasets.SplitGenerator(
                  name=datasets.Split.TEST,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "test-" + self.config.name + ".jsonl")},
                    )
              ),
          return splits


          data_dir = 'download/xcopa'
          splits =  [
              datasets.SplitGenerator(
                  name=datasets.Split.VALIDATION,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "val-" + self.config.name + ".jsonl")},
              ),
          ]
          if self.config.name == 'en':
            splits.append(
                datasets.SplitGenerator(
                  name=datasets.Split.TRAIN,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "train-" + self.config.name + ".jsonl")},
                    )
                )
          else:
            splits.append(
              datasets.SplitGenerator(
                  name=datasets.Split.TEST,
                  # These kwargs will be passed to _generate_examples
                  gen_kwargs={"filepath": os.path.join(data_dir, "test-" + self.config.name + ".jsonl")},
                    )
              ),
          return splits

    def _generate_examples(self, filepath):
        """Yields examples."""
        if isinstance(filepath, list):
            count = 0
            for path_ in filepath:
              with open(path_, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    if "changed" not in data:  # Does not exist in English COPA
                      data["changed"] = False
                    idx = data["idx"]
                    yield count, data
                    count += 1
        else:
          with open(filepath, encoding="utf-8") as f:
              for row in f:
                  data = json.loads(row)
                  if "changed" not in data:  # Does not exist in English COPA
                    data["changed"] = False
                  idx = data["idx"]
                  yield idx, data
