# coding=utf-8
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
"""Testing code for evaluate.py."""

import collections
import os

from absl.testing import absltest
from absl.testing import parameterized
from xtreme.evaluate import evaluate_one_task
from xtreme.evaluate import GROUP2TASK
from xtreme.evaluate import TASK2LANGS

DATA_DIR = './/mock_test_data'

# Mock submission scores for testing
TASK2AVG_SCORES = {
    'pawsx': {'avg_accuracy': 51.42857142857143},
    'xnli': {'avg_accuracy': 30.666666666666668},
    'panx': {'avg_f1': 57.50793650793652, 'avg_precision': 54.729166666666664,
             'avg_recall': 62.750000000000014},
    'udpos': {'avg_f1': 70.21746048354693, 'avg_precision': 71.02232625883823,
              'avg_recall': 69.54982073976082},
    'bucc2018': {'avg_f1': 55.0, 'avg_precision': 55.0, 'avg_recall': 55.0},
    'tatoeba': {'avg_accuracy': 53.611111111111114},
    'xquad': {'avg_exact_match': 77.27272727272727, 'avg_f1': 79.9586776859504},
    'mlqa': {'avg_exact_match': 57.142857142857146, 'avg_f1': 81.76870748299321},
    'tydiqa': {'avg_exact_match': 88.88888888888889, 'avg_f1': 97.22222222222223}
}


class EvaluateTest(parameterized.TestCase):
  """Test cases for evaluate.py."""

  @parameterized.named_parameters(
      ('PAWS-X', 'pawsx'),
      ('XNLI', 'xnli'),
      ('PANX', 'panx'),
      ('UDPOS', 'udpos'),
      ('BUCC2018', 'bucc2018'),
      ('Tatoeba', 'tatoeba'),
      ('XQuAD', 'xquad'),
      ('MLQA', 'mlqa'),
      ('TyDiQA', 'tydiqa'))
  def testTask(self, task):
    data_dir = os.path.join(absltest.get_default_test_srcdir(), DATA_DIR)
    suffix = 'json' if task in GROUP2TASK['qa'] else 'tsv'
    score = collections.defaultdict(dict)
    for lg in TASK2LANGS[task]:
      pred_file = os.path.join(data_dir, 'predictions', task,
                               f'test-{lg}.{suffix}')
      label_file = os.path.join(data_dir, 'labels', task, f'test-{lg}.{suffix}')
      score_lg = evaluate_one_task(pred_file, label_file, task, language=lg)
      for metric in score_lg:
        score[metric][lg] = score_lg[metric]
    avg_score = {}
    for m in score:
      avg_score[f'avg_{m}'] = sum(score[m].values()) / len(score[m])
    self.assertEqual(avg_score, TASK2AVG_SCORES[task])


if __name__ == '__main__':
  absltest.main()
