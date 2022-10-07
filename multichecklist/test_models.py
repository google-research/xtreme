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
"""Main script to evaluate a model on multilingual CheckList tests."""
import csv
import os

from absl import app
from absl import flags
from checklist.test_suite import TestSuite
from checklist_utils import format_squad_with_context
import numpy as np
import transformers

np.random.seed(42)

FLAGS = flags.FLAGS
flags.DEFINE_string('tests_dir', 'lang_suites',
                    'The path to the directory of the saved test files.')
flags.DEFINE_string('pred_dir', 'lang_predictions',
                    'The path to the directory where predictions are saved.')
flags.DEFINE_string('out_file', 'results_table.csv',
                    'The file where a table of the results should be saved.')
flags.DEFINE_enum('model_name', 'xlmr', ['xlmr', 'mbert'],
                  'The fine-tuned QA model that should be used.')
flags.DEFINE_string('model_path', None,
                    'The path to the fine-tuned QA model.')

TEST_NAME2SHORT = {'A is COMP than B. Who is more / less COMP?': 'comp',
                   'Intensifiers (very, super, extremely) and reducers '
                   '(somewhat, kinda, etc)?': 'intense',
                   'size, shape, age, color': 'props',
                   'Profession vs nationality': 'profession_nationality',
                   'Animal vs Vehicle': 'animal_vehicle',
                   'Animal vs Vehicle v2': 'animal_vehicle_v2'}


def main(_):

  if not os.path.isdir(FLAGS.tests_dir):
    raise FileNotFoundError(f'{FLAGS.tests_dir} does not exist.')
  if not os.path.isdir(FLAGS.pred_dir):
    print(f'Creating directory {FLAGS.pred_dir}.')
    os.mkdir(FLAGS.pred_dir)

  languages = sorted(
      list(set([f_name[:2] for f_name in os.listdir(FLAGS.tests_dir)])))

  # If no model is provided, we evaluate the predictions in lang_predictions
  if FLAGS.model_path is None:
    print('Evaluating results based on written predictions.')
  else:
    if FLAGS.model_name == 'mbert':
      config = transformers.BertConfig.from_pretrained(FLAGS.model_path)
      tokenizer = transformers.BertTokenizer.from_pretrained(
          FLAGS.model_path, do_lower_case=False)
      transformer_model = transformers.BertForQuestionAnswering.from_pretrained(
          FLAGS.model_path, config=config)
    elif FLAGS.model_name == 'xlmr':
      config = transformers.XLMRobertaConfig.from_pretrained(FLAGS.model_path)
      tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(
          FLAGS.model_path, do_lower_case=False)
      transformer_model = transformers.XLMRobertaForQuestionAnswering.from_pretrained(
          FLAGS.model_path,
          config=config)
    else:
      raise ValueError(f'{FLAGS.model_name} is not available.')

    model = transformers.QuestionAnsweringPipeline(
        transformer_model, tokenizer, device=0)

  def predconfs(context_question_pairs, model, tokenizer):
    preds = []
    confs = []
    contexts, questions = zip(*context_question_pairs)
    # For some examples, XLM-R would predict an invalid start or end index
    # We have modified the dictionary here: https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/question_answering.py#L356
    # to default either to the first or last index respectively
    predictions = model(question=questions, context=contexts)

    for pred in predictions:
      if lang in ['zh', 'ja', 'ko']:
        # To deal with non-whitespace tokenization, we needed to modify the
        # output function in pipelines.py to also output feature.tokens as well
        # as the original predicted start and end positions (s, e). These are
        # added with the corresponding keys to the answer dictionary here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/question_answering.py#L356
        orig_start, orig_end = pred['orig_start'], pred['orig_end']
        answer = tokenizer.convert_tokens_to_string(
            pred['feature_tokens'][orig_start:orig_end+1])
      else:
        answer = pred['answer']
      preds.append(answer)
      confs.append(pred['score'])
    return preds, np.array(confs)

  # Load the suites from file
  lang2suite = {}
  for lang in languages:
    print(f'Reading {lang} suite from file.')
    lang2suite[lang] = TestSuite.from_file(os.path.join(
        FLAGS.tests_dir, f'{lang}_squad_suite.pkl'))

  # Keep track of the error rate for each test and language to generate a table
  lang2test2error = {}

  for lang in languages:
    print(f'\n\n===== Evaluating on {lang} suite. =====')
    suite = lang2suite[lang]
    lang2test2error[lang] = {}

    for test_name, test in suite.tests.items():
      test_short = TEST_NAME2SHORT[test_name]
      pred_file = os.path.join(
          FLAGS.pred_dir, f'{lang}_{test_short}_{FLAGS.model_name}_preds.txt')
      print(f'\n----- {test_name} -----')
      examples, _ = test.example_list_and_indices()

      if os.path.exists(pred_file):
        print(f'Reading predictions from {pred_file}')
        test.run_from_file(pred_file, file_format='pred_only')
      elif not FLAGS.model_path:
        continue
      else:
        print(f'Predicting on {len(examples)} {lang} examples.')
        preds, confs = predconfs(examples, model, tokenizer)

        print(f'Writing {lang} predictions to {pred_file}.')
        with open(pred_file, 'w') as f:
          for pred in preds:
            f.write(pred + '\n')
        test.run_from_preds_confs(preds, confs, overwrite=True)
      test.summary(format_example_fn=format_squad_with_context)
      passed = test.results['passed']
      accuracy = len([1 for x in passed if x]) / len(passed)
      error = 1.0 - accuracy
      lang2test2error[lang][test_name] = error

  # Generate a table of the results as a csv
  with open(FLAGS.out_file, 'w') as f:
    writer = csv.writer(f)
    tests = list(TEST_NAME2SHORT.keys())
    columns = ['Lang.'] + tests
    writer.writerow(columns)
    for lang, test2error in lang2test2error.items():
      errors = [test2error[t] * 100 for t in tests]
      writer.writerow([lang] + errors)


if __name__ == '__main__':
  app.run(main)
