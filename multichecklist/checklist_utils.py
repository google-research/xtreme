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
# coding=utf-8
# Based on functions from:
# https://github.com/marcotcr/checklist/blob/master/notebooks/SQuAD.ipynb
# Licensed under the MIT License.
"""Utility functions for working with CheckLists."""

import itertools


def format_squad_with_context(x, pred, conf, label=None, *args, **kwargs):
  c, q = x
  ret = 'C: %s\nQ: %s\n' % (c, q)
  if label is not None:
    ret += 'A: %s\n' % label
  ret += 'P: %s\n' % pred
  return ret


def crossproduct(t):
  # takes the output of editor.template and does the cross product of contexts and qas
  ret = []
  ret_labels = []
  for x in t.data:
    cs = x['contexts']
    qas = x['qas']
    d = list(itertools.product(cs, qas))
    ret.append([(x[0], x[1][0]) for x in d])
    ret_labels.append([x[1][1] for x in d])
  t.data = ret
  t.labels = ret_labels
  return t
