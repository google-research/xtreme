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
"""Data processing for Social IQa.

Based on https://github.com/huggingface/datasets/blob/master/datasets/xcopa/xcopa.py
"""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets


_DESCRIPTION = """\
  Social IQa
"""

LETTER2LABEL = {
    'A': 0, 'B': 1, 'C': 2
}

_LANG = ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]

# Set this to the data directory in your environment
DATA_DIR = '../../download'

class SiqaConfig(datasets.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, **kwargs):
        """

        Args:
            data_dir: directory for the given language dataset
            **kwargs: keyword arguments forwarded to super.
        """
        super(SiqaConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Siqa(datasets.GeneratorBasedBuilder):
    """Social IQa dataset class."""

    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        SiqaConfig(
            name=lang,
            description="Social IQa"
        )
        for lang in [f'translate_train_{l}' for l in _LANG] + ['translate_train_all'] + ['en']
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
                    "context": datasets.Value("string"),
                    "answerA": datasets.Value("string"),
                    "answerB": datasets.Value("string"),
                    "answerC": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "correct": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs

        name = 'socialIQa_v1.4'

        if self.config.name is not None and self.config.name.startswith('translate_train_all'):
          data_dir = os.path.join(DATA_DIR, 'siqa_translate_train')
          gold_dir = os.path.join(DATA_DIR, 'siqa')

          return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(gold_dir, f"{name}_dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": [os.path.join(data_dir, f"{name}_{lang}_translate_train.jsonl") for lang in _LANG] + [os.path.join(gold_dir, f"{name}_trn.jsonl")]},
            ),
            ]

        if self.config.name is not None and self.config.name.startswith('translate_train'):
          data_dir = os.path.join(DATA_DIR, 'siqa_translate_train')
          gold_dir = os.path.join(DATA_DIR, 'siqa')

          lang = self.config.name.split('_')[-1]

          return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, f"{name}_{lang}_translate_dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, f"{name}_{lang}_translate_train.jsonl")},
            ),
            ]

        data_dir = os.path.join(DATA_DIR, 'siqa')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, f"{name}_dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, f"{name}_trn.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, f"{name}_tst.jsonl")},
            )
            ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(xcopa): Yields (key, example) tuples from the dataset
        if isinstance(filepath, list):
          count = 0
          for path_ in filepath:
            with open(path_, encoding="utf-8") as f:
              for i, row in enumerate(f):
                  data = json.loads(row)
                  data["label"] = LETTER2LABEL[data["correct"]]
                  if 'idx' in data:
                    data.pop('idx')
                  yield count, data
                  count += 1
        else:

          with open(filepath, encoding="utf-8") as f:
              for i, row in enumerate(f):
                  data = json.loads(row)
                  data["label"] = LETTER2LABEL[data["correct"]]
                  if 'idx' in data:
                    data.pop('idx')
                  yield i, data
