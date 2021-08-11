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
"""Evaluation."""

import argparse
import collections
import json
import os
import sys

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from third_party.evaluate_mlqa import evaluate as mlqa_eval
from third_party.evaluate_squad import evaluate as squad_eval


def read_tag(file):
  """Read labels of NER and POS data."""
  labels = []
  example = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        example.append(line)
      else:
        labels.append(example)
        example = []
  if example:
    labels.append(example)
  return labels


def read_label(file):
  with open(file, 'r') as f:
    return [l.strip() for l in f]


def read_squad(file):
  """Read QA data."""
  with open(file) as dataset_file:
    dataset_json = json.load(dataset_file)
    if 'data' in dataset_json:
      return dataset_json['data']
    else:
      return dataset_json


def f1(labels, predictions, language=None):
  f1_val = f1_score(labels, predictions)
  prec = precision_score(labels, predictions)
  rec = recall_score(labels, predictions)
  return {'f1': f1_val * 100, 'precision': prec * 100, 'recall': rec * 100}


def accuracy(labels, predictions, language=None):
  correct = sum([int(p == l) for p, l in zip(predictions, labels)])
  accuracy_score = float(correct) / len(predictions)
  return {'accuracy': accuracy_score * 100}


def bucc_f1(labels, predictions, language=None):
  """Calculate F1 score for BUCC data."""
  labels = set([tuple(l.split('\t')) for l in labels])
  predictions = set([tuple(l.split('\t')) for l in predictions])
  ncorrect = len(labels.intersection(predictions))
  if ncorrect > 0:
    prec = ncorrect / len(predictions)
    rec = ncorrect / len(labels)
    f1_val = 2 * prec * rec / (prec + rec)
  else:
    prec = rec = f1_val = 0
  return {'f1': f1_val * 100, 'precision': prec * 100, 'recall': rec * 100}


def squad_em_f1(labels, predictions, language=None):
  return squad_eval(labels, predictions)


def mlqa_em_f1(labels, predictions, language):
  if language is None:
    print('required 2-char language code for the argument `language`')
    sys.exit(0)
  return mlqa_eval(labels, predictions, language)


GROUP2TASK = {
    'classification': ['pawsx', 'xnli'],
    'tagging': ['udpos', 'panx'],
    'qa': ['xquad', 'mlqa', 'tydiqa'],
    'retrieval': ['bucc2018', 'tatoeba'],
}


TASK2LANGS = {
    'pawsx': 'de,en,es,fr,ja,ko,zh'.split(','),
    'xnli': 'ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh'.split(','),
    'panx': 'ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,'
            'it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu'.split(','),
    'udpos': 'af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,'
             'nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh'.split(','),
    'bucc2018': 'de,fr,ru,zh'.split(','),
    'tatoeba': 'ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,'
               'pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu'.split(','),
    'xquad': 'en,es,de,el,ru,tr,ar,vi,th,zh,hi'.split(','),
    'mlqa': 'en,es,de,ar,hi,vi,zh'.split(','),
    'tydiqa': 'en,ar,bn,fi,id,ko,ru,sw,te'.split(','),
}


READER_FUNCTION = {
    'pawsx': read_label,
    'xnli': read_label,
    'panx': read_tag,
    'udpos': read_tag,
    'bucc2018': read_label,
    'tatoeba': read_label,
    'xquad': read_squad,
    'mlqa': read_squad,
    'tydiqa': read_squad,
}


METRIC_FUNCTION = {
    'pawsx': accuracy,
    'xnli': accuracy,
    'panx': f1,
    'udpos': f1,
    'bucc2018': bucc_f1,
    'tatoeba': accuracy,
    'xquad': squad_em_f1,
    'mlqa': mlqa_em_f1,
    'tydiqa': squad_em_f1,
}


def evaluate_one_task(prediction_file, label_file, task, language=None):
  r"""Evaluate the classification tasks by accuracy.

  Args:
    prediction_file (string): path to the prediction tsv file.
    label_file (string): path to the grouth truth tsv file.
    task (string): task identifier
    language (string): language ISO code
  Returns:
    result (dict): a dictionary with accuracy.

  Both input files contain one example per line as follows:
    ``[label]\t[sentence1]\t[sentence2]``
  """
  predictions = READER_FUNCTION[task](prediction_file)
  labels = READER_FUNCTION[task](label_file)
  if task not in ['bucc2018', 'mlqa', 'tydiqa', 'xquad']:
    assert len(predictions) == len(labels), (
        'Number of examples in {} and {} not matched in {} task'.format(
            prediction_file, label_file, task))
  result = METRIC_FUNCTION[task](labels, predictions, language)
  return result


def evaluate(prediction_folder, label_folder, verbose=False):
  """Evaluate on all tasks if available.

  Args:
    prediction_folder (string): prediction folder that contains each task's
                                prediction in each subfolder.
    label_folder (string): label folder that contains each task's ground-truth
                           label in each subfolder.
    verbose (boolean): whether to print average results during evaluation.
  Returns:
    overall_scores (dict): a dictionary with sub-group scores. key: group label.
    detailed_scores (dict): a dictionary with detailed scores. key: task label.
  """
  prediction_tasks = next(os.walk(prediction_folder))[1]
  label_tasks = next(os.walk(label_folder))[1]

  detail_scores = {}
  for task, langs in TASK2LANGS.items():
    if task in prediction_tasks and task in label_tasks:
      suffix = 'json' if task in GROUP2TASK['qa'] else 'tsv'
      # collect scores over all languages
      score = collections.defaultdict(dict)
      for lg in langs:
        prediction_file = os.path.join(
            prediction_folder, task, f'test-{lg}.{suffix}')
        label_file = os.path.join(label_folder, task, f'test-{lg}.{suffix}')
        score_lg = evaluate_one_task(
            prediction_file, label_file, task, language=lg)
        for metric in score_lg:
          score[metric][lg] = score_lg[metric]
      # average over all languages
      avg_score = {}
      for m in score:
        avg_score[f'avg_{m}'] = sum(score[m].values()) / len(score[m])
      score.update(avg_score)
      if task in GROUP2TASK['qa']:
        score['avg_metric'] = (score['avg_exact_match'] + score['avg_f1']) / 2
      elif 'avg_f1' in score:
        score['avg_metric'] = score['avg_f1']
      elif 'avg_accuracy' in score:
        score['avg_metric'] = score['avg_accuracy']
      detailed_scores[task] = score
      if verbose:
        avg_result = ', '.join(['{}={:.1f}'.format(k, v)
                                for k, v in score.items()
                                if k.startswith('avg')])
        print('- Evaluate {}:\t{}'.format(task, avg_result))

  # Display logic:
  overall_scores = {}
  all_tasks = set(TASK2LANGS.keys())
  available_tasks = set(detail_scores.keys())

  # If scores of all tasks are available, show overall score in the main table
  if all_tasks == available_tasks:
    overall_scores['all_task'] = sum(detailed_scores[task]['avg_metric']
                                     for task in all_tasks) / len(all_tasks)

  # If scores of all tasks in a group are available, show score in the sub table
  for group, group_tasks in GROUP2TASK.items():
    if not set(group_tasks) - available_tasks:
      overall_scores[group] = sum(detailed_scores[task]['avg_metric']
                                  for task in group_tasks) / len(group_tasks)

  return overall_scores, detailed_scores


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--prediction_folder', default=None, type=str,
                      required=True, help='the predictions of one model')
  parser.add_argument('--label_folder', default=None, type=str, required=True,
                      help='the grouth truth file')
  parser.add_argument('--verbose', action='store_true', default=False,
                      help='whether to print details')
  args = parser.parse_args()
  group_scores, detailed_scores = evaluate(
      args.prediction_folder, args.label_folder, args.verbose)
  group_scores.update(detailed_scores)
  for task_name, task_dict in group_scores.items():
    print('====== %s ======\n' % task_name)
    metrics = []
    for metric_name, metric_dict in task_dict.items():
      if metric_name.startswith('avg'):
        continue
      print('------ %s ------' % metric_name)
      languages, scores = [], []
      for lang, score_value in metric_dict.items():
        languages.append(lang)
        scores.append('%.2f' % score_value)
      print(', '.join(languages))
      print(', '.join(scores))
      print()
      metrics.append(metric_name)
    metrics.append('metric')
    for metric_name in metrics:
      avg_score_value = task_dict['avg_%s' % metric_name]
      print('%s: %.2f' % (metric_name, avg_score_value))
