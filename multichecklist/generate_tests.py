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
"""Main script to read CheckList templates and generate multilingual tests."""
import collections
import os

from absl import app
from absl import flags
import checklist
import checklist.editor
import checklist.test_suite
import generate_test_utils
import numpy as np

np.random.seed(42)


FLAGS = flags.FLAGS
flags.DEFINE_string('templates_path', 'checklist_templates.tsv',
                    'The path to the templates tsv file.')
flags.DEFINE_string('save_dir', 'lang_suites',
                    'The directory where the CheckList suites should be saved.')
flags.DEFINE_integer('num_test_samples', 200,
                     'The number of test cases to generate for each test.')


class TestEntry:
  """Class for a templated test."""

  def __init__(self, name, capability, contexts, questions, answers, args):
    self.name = name
    self.capability = capability
    self.contexts = contexts
    self.questions = questions
    self.answers = answers
    self.args = args

  def __str__(self):
    return f'Name: {self.name}. Capability: {self.capability}.'


def read_translations(file_path):
  """Read the multilingual CheckList templates from a file."""
  # Columns are named with language iso codes, e.g. en, af, etc. For some
  # languages where multiple translations are possible in some cases, we include
  # multiple columns. These are of the format [LANG], [LANG]-placeholders,
  # [LANG], [LANG]-placeholders etc. The [LANG] column contain varying
  # translations while the placeholders columns contain the argument values
  # for which the translations vary.
  lang2name2test = collections.defaultdict(dict)
  with open(file_path, 'r', encoding='utf-8') as f:
    name, capability = None, None
    lang2contexts = collections.defaultdict(list)
    lang2questions = collections.defaultdict(list)
    lang2answers = collections.defaultdict(list)
    columns = None
    lang2args = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, line in enumerate(f):
      line = line.strip()
      if i == 0:
        columns = line.split('\t')[1:]
      elif line.startswith('Name'):
        name = line.split('\t')[1]
      elif line.startswith('Capability'):
        capability = line.split('\t')[1]
      elif line.startswith('Context'):
        translations = line.split('\t')[1:]
        for lang, translation in zip(columns, translations):
          if name in generate_test_utils.GENDER_MARKING_TEMPLATES:
            # For tests where adjectives and nouns may need to be declined,
            # templates only contain the male version so we restrict replacement
            translation = translation.replace('first_name', 'male')
          if not lang.endswith('placeholders'):
            # We generally also add empty strings to keep alignment
            lang2contexts[lang].append(translation)
          else:
            lang = lang.split('-')[0]
            lang2args[lang]['contexts_placeholders'].append(translation)
      elif line.startswith('Question'):
        translations = line.split('\t')[1:]
        for lang, translation in zip(columns, translations):
          if name in generate_test_utils.GENDER_MARKING_TEMPLATES:
            translation = translation.replace('first_name', 'male')
          if not lang.endswith('placeholders'):
            lang2questions[lang].append(translation)
          else:
            lang = lang.split('-')[0]
            lang2args[lang]['questions_placeholders'].append(translation)
      elif line.startswith('Answer'):
        translations = line.split('\t')[1:]
        for lang, translation in zip(columns, translations):
          if name in generate_test_utils.GENDER_MARKING_TEMPLATES:
            translation = translation.replace('first_name', 'male')
          if not lang.endswith('placeholders'):
            # Only for answers we do not add empty strings
            if translation.strip():
              lang2answers[lang].append(translation)
          # We only have answer variations for Quechua
          elif (lang.endswith('placeholders') and lang == 'qu-placeholders' and
                name == 'Animal vs Vehicle'):
            lang = lang.split('-')[0]
            lang2args[lang]['answer_placeholders'].append(translation)
      elif line:
        arg_name, translations = line.split('\t')[0], line.split('\t')[1:]
        for lang, translation in zip(columns, translations):
          if not lang.endswith('placeholders'):
            if (translation.strip() or name == 'size, shape, age, color' or
                name == 'Profession vs nationality'):
              # Only for these tests add empty values to keep alignment with
              # placeholders
              lang2args[lang][arg_name].append(translation)
          elif lang.endswith('placeholders'):
            lang = lang.split('-')[0]
            lang2args[lang][arg_name + '_placeholders'].append(translation)
      else:
        if name is None:
          continue
        assert capability is not None
        assert lang2contexts
        assert lang2questions
        assert lang2answers

        for lang in lang2contexts.keys():
          lang2name2test[lang][name] = TestEntry(
              name, capability, lang2contexts[lang], lang2questions[lang],
              lang2answers[lang], lang2args[lang])
        name, capability = None, None
        lang2contexts = collections.defaultdict(list)
        lang2questions = collections.defaultdict(list)
        lang2answers = collections.defaultdict(list)
        lang2args = collections.defaultdict(
            lambda: collections.defaultdict(list))
    if name is not None:
      assert capability is not None
      assert lang2contexts
      assert lang2questions
      assert lang2answers
      for lang in lang2contexts.keys():
        lang2name2test[lang][name] = TestEntry(
            name, capability, lang2contexts[lang], lang2questions[lang],
            lang2answers[lang], lang2args[lang])
  return lang2name2test


def main(_):
  if not os.path.exists(FLAGS.templates_path):
    raise FileNotFoundError(f'{FLAGS.templates_path} does not exist.')
  if not os.path.isdir(FLAGS.save_dir):
    print(f'Creating directory {FLAGS.save_dir}.')
    os.mkdir(FLAGS.save_dir)

  # Maps from language ISO codes to test names to TestEntry objects
  lang2name2test = read_translations(FLAGS.templates_path)
  languages = list(lang2name2test.keys())

  lang2suite = {}
  for lang in languages:
    lang2suite[lang] = checklist.test_suite.TestSuite()

  for test_name in generate_test_utils.TEST_NAMES:
    test_func = generate_test_utils.name2test_func(test_name)
    print(f'\n{test_name}')
    for lang in languages:
      editor = checklist.editor.Editor(lang)
      t_entry = lang2name2test[lang][test_name]
      en_entry = lang2name2test['en'][test_name]
      if test_name in generate_test_utils.ANIMAL_TEST_NAMES:
        default_entry = lang2name2test[lang][generate_test_utils.ANIMAL_NAME]
        default_en_entry = lang2name2test['en'][generate_test_utils.ANIMAL_NAME]
        test = test_func(lang, editor, t_entry, en_entry, default_entry,
                         default_en_entry, test_name, FLAGS.num_test_samples)
      else:
        test = test_func(lang, editor, t_entry, en_entry,
                         FLAGS.num_test_samples)
      lang2suite[lang].add(test, overwrite=True)
      print('%s (%d tests; %d samples): %s'
            % (lang, len(test.data), len(test.to_raw_examples()),
               ', '.join(test.to_raw_examples()[:2])))

  for lang in languages:
    print(f'Writing {lang} suite to directory {FLAGS.save_dir}.')
    lang2suite[lang].save(os.path.join(FLAGS.save_dir,
                                       f'{lang}_squad_suite.pkl'))
    format_fn = lambda x: {'passage': x[0], 'question': x[1]}
    lang2suite[lang].to_raw_file(os.path.join(FLAGS.save_dir,
                                              f'{lang}_squad_suite.json'),
                                 format_fn=format_fn, file_format='squad')


if __name__ == '__main__':
  app.run(main)
