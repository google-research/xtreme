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
"""Functions for generating test cases from CheckList templates."""

import collections
import functools
import itertools
import re

from checklist.expect import Expect
from checklist.test_types import MFT
from checklist_utils import crossproduct
import munch


COMPARISON_NAME = 'A is COMP than B. Who is more / less COMP?'
INTENSIFIERS_NAME = ('Intensifiers (very, super, extremely) and reducers '
                     '(somewhat, kinda, etc)?')
PROPERTIES_NAME = 'size, shape, age, color'
PROFESSION_NAME = 'Profession vs nationality'
ANIMAL_NAME = 'Animal vs Vehicle'
ANIMAL2_NAME = 'Animal vs Vehicle v2'
TEST_NAMES = [COMPARISON_NAME, INTENSIFIERS_NAME, PROPERTIES_NAME,
              PROFESSION_NAME, ANIMAL_NAME, ANIMAL2_NAME]
ANIMAL_TEST_NAMES = [ANIMAL_NAME, ANIMAL2_NAME]
GENDER_MARKING_TEMPLATES = [COMPARISON_NAME, INTENSIFIERS_NAME, PROFESSION_NAME]

NUM_TEST_SAMPLES = 200


def name2test_func(name):
  """Maps from the test name to the respective function to generate tests."""
  if name == COMPARISON_NAME:
    return generate_comparison_tests
  if name == INTENSIFIERS_NAME:
    return generate_intensifiers_tests
  if name == PROPERTIES_NAME:
    return generate_properties_tests
  if name == PROFESSION_NAME:
    return generate_profession_tests
  if name in ANIMAL_TEST_NAMES:
    return generate_animal_tests
  raise ValueError(f'{name} is not recognized.')


def clean(text, lang):
  """Utility function for normalizing answers."""
  articles = ''
  punctuation = ',. '
  if lang == 'af':
    articles = r'(’n)\b'
  if lang == 'ar':
    articles = r'\sال^|ال'
  if lang == 'bn':
    articles = r'\b(একটি|একজন)'  # not separated by a word boundary
    punctuation += '৷'  # full stop
  if lang == 'de':
    articles = r'\b(ein|eine|einen)\b'
  if lang == 'el':
    articles = r'\b(Ο|ένα|μια)\b'
  if lang == 'en':
    articles = r'\b(a|an|the|in|at)\b'
  if lang == 'es':
    articles = r'\b(un|una|unos|unas|el|la|los|las)\b'
  if lang == 'eu':
    articles = r'\b(bat)\b'
  if lang == 'fa':
    articles = r'\b(یک)\b'
  if lang == 'fr':
    articles = r'\b(un|une|le|la|en)\b'
  if lang == 'hi':
    articles = r'\b(एक)'
    punctuation += '৷'  # full stop
  if lang == 'hu':
    articles = r'\b(egy)\b'
  if lang == 'it':
    articles = r'\b(un|una)\b'
  if lang == 'ml':
    articles = r'\b(ഒരു)'
  if lang == 'mr':
    articles = r'\b(एक)'
  if lang == 'my':
    articles = r'\b(တစ်ဦး)'
  if lang == 'nl':
    articles = r'\b(een)\b'
  if lang == 'pa':
    articles = r'\b(ਇੱਕ|ਇਕ)'
    punctuation += '।'  # full stop
  if lang == 'pl':
    punctuation = '„”,. '
  if lang == 'pt':
    articles = r'\b(um|uma)\b'
  if lang == 'ru':
    punctuation += ','
  if lang == 'ta':
    articles = r'\b(ஒரு)'
  if lang == 'tl':
    articles = r'\b(Si)\b'
  if lang == 'tr':
    articles = r'\b(bir)\b'
  if lang == 'ur':
    punctuation += '\u200f'  # right-to-left-mark
  if lang == 'vi':
    articles = r'\b(một)\b'
  if lang == 'wo':
    articles = r'\b(dafa)\b'  # this is a pronoun
  if lang == 'yo':
    articles = r'\b(kan)\b'
  return re.sub(articles, ' ', text).strip(punctuation)


def expect_squad(x, pred, conf, label=None, meta=None, lang=None):
  """Function for comparing a prediction and label."""
  return clean(pred, lang) == clean(label, lang)


def generate_comparison_tests(lang, editor, t_entry, en_entry,
                              num_test_samples=NUM_TEST_SAMPLES):
  """Method to generate comparison tests.

  Args:
    lang: the language ISO code
    editor: a checklist.editor.Editor instance
    t_entry: the TestEntry instance of the test in the specified language
    en_entry: the TestEntry instance of the test in English
    num_test_samples: the number of test cases to generate
  Returns:
    a checklist.test_types.MFT instance
  """
  sep = ', '
  if lang in ['my', 'th']:
    sep = ' '
  elif lang in ['ur', 'ar', 'fa']:
    sep = '،'
  elif lang in ['zh']:
    sep = '，'

  if lang in ['ja', 'ta', 'ru', 'hu', 'pt']:
    placeholders2dict = {}
    for question, placeholders in zip(
        t_entry.questions, t_entry.args['questions_placeholders']):
      if placeholders not in placeholders2dict:
        placeholders2dict[placeholders] = {'questions': [], 'adj': []}
      placeholders2dict[placeholders]['questions'].append(question)
      for adj, en_adjs in zip([e for e in t_entry.args['adj'] if e],
                              en_entry.args['adj']):
        en_adj = en_adjs.split(',')[1]
        # Add the adjective to the corresponding placeholders if its
        # English translation is listed
        if not placeholders or en_adj in placeholders:
          placeholders2dict[placeholders]['adj'].append(adj)

    # Add the question that applies to all adjectives to every setting
    if '' in placeholders2dict:
      questions = placeholders2dict.pop('')['questions']
      for p_dict in placeholders2dict.values():
        p_dict['questions'] += [q for q in questions if q]

    t = None
    nsamples = num_test_samples // len(placeholders2dict.keys())
    for placeholders, p_dict in placeholders2dict.items():
      tests = create_comparison_crossproduct(
          editor, [c for c in t_entry.contexts if c], p_dict['questions'],
          t_entry.answers, p_dict['adj'], sep, nsamples)
      if t is None:
        t = tests
      else:
        t += tests
  else:
    t = create_comparison_crossproduct(
        editor, [c for c in t_entry.contexts if c],
        [q for q in t_entry.questions if q], t_entry.answers,
        t_entry.args['adj'], sep, num_test_samples)

  return MFT(**t, name=COMPARISON_NAME, expect=Expect.single(
      functools.partial(expect_squad, lang=lang)),
             capability=t_entry.capability)


def create_comparison_crossproduct(editor, contexts, questions, answers, adj,
                                   sep=', ', nsamples=NUM_TEST_SAMPLES):
  """Method for creating the crossproduct for comparison tests."""
  return crossproduct(editor.template(
      {
          'contexts': contexts,
          'qas': [(q, a) for q, a in zip(questions, answers)]
      },
      save=True,
      adj=[adj.split(sep) for adj in adj],
      remove_duplicates=True,
      nsamples=nsamples,
  ))


def generate_intensifiers_tests(lang, editor, t_entry, en_entry,
                                num_test_samples=NUM_TEST_SAMPLES):
  """Method to generate intensifier tests.

  Args:
    lang: the language ISO code
    editor: a checklist.editor.Editor instance
    t_entry: the TestEntry instance of the test in the specified language
    en_entry: the TestEntry instance of the test in English
    num_test_samples: the number of test cases to generate
  Returns:
    a checklist.test_types.MFT instance
  """
  if lang in ['de', 'ru', 'ja', 'vi']:
    placeholders2dict = {}
    for state, en_state in zip(t_entry.args['state'],
                               en_entry.args['state']):
      placeholders2dict[en_state] = {
          'questions': [], 'contexts': [], 'state': [state], 'very': [],
          'somewhat': []}

    for question, question_placeholders in zip(
        t_entry.questions, t_entry.args['questions_placeholders']):
      if question_placeholders:
        for placeholder in question_placeholders.split(', '):
          placeholders2dict[placeholder]['questions'].append(question)

    for context, context_placeholders in zip(
        t_entry.contexts, t_entry.args['contexts_placeholders']):
      if context_placeholders:
        for placeholder in context_placeholders.split(', '):
          placeholders2dict[placeholder]['contexts'].append(context)

    for mod in ['very', 'somewhat']:
      for mod_arg, mod_placeholders in zip(
          t_entry.args[mod], t_entry.args[f'{mod}_placeholders']):
        if not mod_placeholders and mod_arg:
          # If there is no placeholder, add the modifier to every state
          for p_dict in placeholders2dict.values():
            p_dict[mod].append(mod_arg)
        elif mod_placeholders:
          for placeholder in mod_placeholders.split(', '):
            placeholders2dict[placeholder][mod].append(mod_arg)

    t = None
    nsamples = num_test_samples // len(placeholders2dict.keys())
    for placeholders, p_dict in placeholders2dict.items():
      tests = create_intensifier_crossproduct(
          editor, p_dict['contexts'], p_dict['questions'], t_entry.answers,
          p_dict['state'], t_entry.args['very'], t_entry.args['somewhat'],
          nsamples)
      if t is None:
        t = tests
      else:
        t += tests
  else:
    t = create_intensifier_crossproduct(
        editor, [c for c in t_entry.contexts if c],
        [q for q in t_entry.questions if q], t_entry.answers,
        t_entry.args['state'], t_entry.args['very'],
        t_entry.args['somewhat'], num_test_samples)

  # Each test consists of 12 QA pairs (same as in the original CheckList)
  return MFT(**t, name=INTENSIFIERS_NAME, expect=Expect.single(
      functools.partial(expect_squad, lang=lang)),
             capability=t_entry.capability)


def create_intensifier_crossproduct(editor, contexts, questions, answers,
                                    state, very, somewhat,
                                    nsamples=NUM_TEST_SAMPLES):
  """Method for creating tests about intensifiers."""
  return crossproduct(editor.template(
      {
          'contexts': contexts,
          'qas': [(q, a) for q, a in zip(questions, answers)]
      },
      s=state,
      very=very,
      somewhat=somewhat,
      remove_duplicates=True,
      nsamples=nsamples,
      save=True
  ))


def generate_properties_tests(lang, editor, t_entry, en_entry,
                              num_test_samples=NUM_TEST_SAMPLES):
  """Method to generate tests about properties.

  Args:
    lang: the language ISO code
    editor: a checklist.editor.Editor instance
    t_entry: the TestEntry instance of the test in the specified language
    en_entry: the TestEntry instance of the test in English
    num_test_samples: the number of test cases to generate
  Returns:
    a checklist.test_types.MFT instance
  """
  sep = ', '
  if lang in ['my', 'th']:
    sep = ' '
  elif lang in ['ur', 'ar', 'fa']:
    sep = '،'
  elif lang in ['zh']:
    sep = '，'

  t_properties = [p for p in t_entry.args['property'] if p]
  en_properties = en_entry.args['property']
  en_prop2t_prop = {en_prop: t_prop for en_prop, t_prop
                    in zip(en_properties, t_properties)}

  contexts = [c for c in t_entry.contexts if c]
  t = None
  if lang not in ['my', 'he', 'th', 'zh', 'hi', 'ja', 'ta', 'ko', 'ml',
                  'mr', 'pa']:
    contexts = [c.replace('{obj[1]}', '{c:obj[1]}') for c in contexts]

  if lang in ['es', 'he', 'et', 'fi', 'fr', 'it', 'uk', 'ur', 'bg', 'nl',
              'hi', 'ru', 'ja', 'fa', 'pl', 'pt', 'el', 'lt', 'de']:
    # Variation in question based on property
    ru_pl_age_question2placeholders = {}
    question2properties_dict = collections.defaultdict(set)
    for question, question_placeholders in zip(
        t_entry.questions, t_entry.args['questions_placeholders']):
      if question_placeholders:
        if lang == 'ru' and question_placeholders.startswith('age'):
          ru_pl_age_question2placeholders[question] = question_placeholders
          question_placeholders = 'age'
        question2properties_dict[question].update(
            question_placeholders.strip().split(', '))
      elif not question_placeholders and question:
        question2properties_dict[question].update(en_properties)

    # Variation in context based on property in Dutch
    nl_property2context = {}
    if lang == 'nl':
      for context, context_placeholders in zip(
          contexts, t_entry.args['contexts_placeholders']):
        for nl_property in context_placeholders.split(', '):
          nl_property2context[nl_property] = context

    # Variation in attribute based on object
    obj_placeholders2dict = {}
    for obj, en_obj in zip(
        [e for e in t_entry.args['obj'] if e], en_entry.args['obj']):
      obj_placeholders2dict[en_obj] = {
          'args': collections.defaultdict(list), 'obj': [obj]}

    for p in en_properties:
      for attribute, attribute_placeholders in zip(
          t_entry.args[p], t_entry.args[p + '_placeholders']):
        for obj, obj_dict in obj_placeholders2dict.items():
          if (attribute_placeholders and obj in attribute_placeholders
              or obj.split(' ')[-1] in attribute_placeholders):
            obj_dict['args'][p].append(attribute)
          elif not attribute_placeholders and attribute:
            # The attribute is valid for all objects
            obj_dict['args'][p].append(attribute)

    nsamples = int(num_test_samples
                   // len(question2properties_dict.keys()) / 10)
    if lang == 'pl':
      nsamples *= 3
    elif lang == 'ru':
      nsamples = 1
    for question1, properties1 in question2properties_dict.items():
      for question2, properties2 in question2properties_dict.items():
        if lang != 'ru':
          if 'p1' not in question1 or 'p2' not in question2:
            continue
        if lang == 'pl' and (
            ('p1' not in question1 and 'age' not in properties1) or
            ('p2' not in question2 and 'age' not in properties2)):
          continue
        if properties1 == properties2 and len(properties1) == 1:
          continue
        # Create property-attribute pairs based on properties and position
        for obj, obj_dict in obj_placeholders2dict.items():
          if (lang in ['ru', 'pl'] and 'age' in properties1
              and obj not in ru_pl_age_question2placeholders[question1]):
            # Skip questions that don't match the object; as we iterate
            # through all questions, we cover all combinations anyway
            continue
          if (lang in ['ru', 'pl'] and 'age' in properties2
              and obj not in ru_pl_age_question2placeholders[question2]):
            continue
          props = []
          for p1 in properties1:
            for p2 in properties2:
              if p1 == p2:
                continue
              for v1, v2 in itertools.product(obj_dict['args'][p1],
                                              obj_dict['args'][p2]):
                props_attributes = {
                    'v1': v1,  # attribute1
                    'v2': v2,  # attribute2
                }
                if lang not in ['pl', 'ru'] or lang == 'pl' and p1 != 'age':
                  # In Russian and Polish, the questions don't explicitly
                  # mention the property
                  props_attributes['p1'] = en_prop2t_prop[p1]
                if lang not in ['pl', 'ru'] or lang == 'pl' and p2 != 'age':
                  props_attributes['p2'] = en_prop2t_prop[p2]
                props.append(munch.Munch(props_attributes))

          if lang == 'nl':
            if 'material' in properties1:
              contexts = [nl_property2context['material']]
            else:
              contexts = [nl_property2context['color']]

          if t is not None and len(t.data) > NUM_TEST_SAMPLES:
            break

          tests = create_attribute_crossproduct(
              editor, contexts, [question1, question2], t_entry.answers,
              obj_dict['obj'], props, sep, nsamples=nsamples)
          if t is None:
            t = tests
          else:
            t += tests

  elif lang == 'mr':
    # Variation in context based on object
    obj_placeholders2dict = {}
    for obj, en_obj in zip([e for e in t_entry.args['obj'] if e],
                           en_entry.args['obj']):
      obj_placeholders2dict[en_obj] = {'contexts': [], 'obj': [obj]}

    for context, contexts_placeholders in zip(
        t_entry.contexts, t_entry.args['contexts_placeholders']):
      for obj, obj_dict in obj_placeholders2dict.items():
        if (contexts_placeholders and obj in contexts_placeholders or
            obj.split(' ')[-1] in contexts_placeholders):
          obj_dict['contexts'].append(context)
        elif not contexts_placeholders and context:
          # The obj is valid for all contexts
          obj_dict['contexts'].append(context)

    nsamples = num_test_samples // len(obj_placeholders2dict.keys())
    for obj, obj_dict in obj_placeholders2dict.items():
      props = create_property_product(t_entry.args, en_prop2t_prop)
      tests = create_attribute_crossproduct(
          editor, obj_dict['contexts'], [q for q in t_entry.questions if q],
          t_entry.answers, obj_dict['obj'], props, sep, nsamples=nsamples)
      if t is None:
        t = tests
      else:
        t += tests
      if len(t.data) > NUM_TEST_SAMPLES:
        break
  else:
    props = create_property_product(t_entry.args, en_prop2t_prop)
    t = create_attribute_crossproduct(
        editor, contexts, [q for q in t_entry.questions if q],
        t_entry.answers, [o for o in t_entry.args['obj'] if o], props, sep,
        num_test_samples)
  return MFT(**t, name=PROPERTIES_NAME, expect=Expect.single(
      functools.partial(expect_squad, lang=lang)),
             capability=t_entry.capability)


def create_property_product(args, en_prop2t_prop):
  """Method for creating tests about properties."""
  props = []
  properties = list(en_prop2t_prop.keys())
  for i in range(len(properties)):
    for j in range(i + 1, len(properties)):
      p1, p2 = properties[i], properties[j]
      for v1, v2 in itertools.product(
          [a for a in args[p1] if a], [a for a in args[p2] if a]):
        props.append(munch.Munch({
            'p1': en_prop2t_prop[p1],  # property1
            'p2': en_prop2t_prop[p2],  # property2
            'v1': v1,  # attribute1
            'v2': v2,  # attribute2
        }))
  return props


def create_attribute_crossproduct(
    editor, contexts, questions, answers, objects, props, separator=', ',
    nsamples=NUM_TEST_SAMPLES):
  """Method for creating tests about attributes."""
  return crossproduct(editor.template(
      {
          'contexts': contexts,
          'qas': [(q, a) for q, a in zip(questions, answers)]
      },
      obj=[obj.split(separator) for obj in objects],
      p=props,
      remove_duplicates=True,
      nsamples=nsamples,
      save=True
    ))


def generate_profession_tests(lang, editor, t_entry, en_entry,
                              num_test_samples=NUM_TEST_SAMPLES):
  """Method to generate tests about professions and nationalities.

  Args:
    lang: the language ISO code
    editor: a checklist.editor.Editor instance
    t_entry: the TestEntry instance of the test in the specified language
    en_entry: the TestEntry instance of the test in English
    num_test_samples: the number of test cases to generate
  Returns:
    a checklist.test_types.MFT instance
  """
  t = None
  if lang in ['fr', 'he', 'it', 'es', 'uk', 'ru', 'pt', 'el', 'lt']:
    # Variation in nationality based on profession
    prof_placeholders2dict = {}
    for prof, en_prof in zip([e for e in t_entry.args['profession'] if e],
                             en_entry.args['profession']):
      prof_placeholders2dict[en_prof] = {
          'nationality': [], 'profession': [prof], 'contexts': [],
          'questions': []}

    for arg_name, arg_values, arg_placeholders_set in [
        ('nationality', t_entry.args['nationality'],
         t_entry.args['nationality_placeholders']),
        ('contexts', t_entry.contexts,
         t_entry.args['contexts_placeholders']),
        ('questions', t_entry.questions,
         t_entry.args['questions_placeholders'])]:
      for arg_value, arg_placeholders in zip(arg_values,
                                             arg_placeholders_set):
        for prof, prof_dict in prof_placeholders2dict.items():
          if (arg_placeholders and prof in arg_placeholders
              or prof.split(' ')[-1] in arg_placeholders):
            prof_dict[arg_name].append(arg_value)
          elif not arg_placeholders and arg_value:
            # The nationality is valid for all professions
            prof_dict[arg_name].append(arg_value)

    nsamples = int(num_test_samples //
                   len(prof_placeholders2dict.keys()) * 1.5)
    for prof, prof_dict in prof_placeholders2dict.items():
      tests = create_profession_crossproduct(
          editor, prof_dict['contexts'], prof_dict['questions'],
          t_entry.answers, prof_dict['nationality'],
          prof_dict['profession'], nsamples=nsamples)
      if t is None:
        t = tests
      else:
        t += tests
  else:
    t = create_profession_crossproduct(
        editor, [c for c in t_entry.contexts if c],
        [q for q in t_entry.questions if q], t_entry.answers,
        [a for a in t_entry.args['nationality'] if a],
        [a for a in t_entry.args['profession'] if a], num_test_samples)
  return MFT(**t, name=PROFESSION_NAME, expect=Expect.single(
      functools.partial(expect_squad, lang=lang)),
             capability=t_entry.capability)


def create_profession_crossproduct(
    editor, contexts, questions, answers, nationalities, professions,
    nsamples=NUM_TEST_SAMPLES):
  """Method for creating tests about professions."""
  return crossproduct(editor.template(
      {
          'contexts': contexts,
          'qas': [(q, a) for q, a in zip(questions, answers)]
      },
      nationality=nationalities,
      profession=professions,
      remove_duplicates=True,
      nsamples=nsamples,
      save=True,
    ))


def generate_animal_tests(lang, editor, t_entry, en_entry, default_entry,
                          default_en_entry, name,
                          num_test_samples=NUM_TEST_SAMPLES):
  """Method to generate tests about animals and vehicles.

  Args:
    lang: the language ISO code
    editor: a checklist.editor.Editor instance
    t_entry: the TestEntry instance of the test in the specified language
    en_entry: the TestEntry instance of the test in English
    default_entry: the TestEntry instance of the first test in the language
    default_en_entry: the TestEntry instance of the first in English
    name: the name of the test (there are two animal tests)
    num_test_samples: the number of test cases to generate

  Returns:
    a checklist.test_types.MFT instance
  """
  t = None
  if lang in ['nl', 'fi']:
    # Variation in contexts based on animal
    animal_placeholders2dict = {}
    for animal, en_animal in zip(
        [e for e in default_entry.args['animal'] if e],
        default_en_entry.args['animal']):
      animal_placeholders2dict[en_animal] = {'contexts': [], 'animal': [animal]}

    for context, contexts_placeholders in zip(
        t_entry.contexts, t_entry.args['contexts_placeholders']):
      for animal, animal_dict in animal_placeholders2dict.items():
        if (contexts_placeholders and animal in contexts_placeholders
            or animal.split(' ')[-1] in contexts_placeholders):
          animal_dict['contexts'].append(context)
        elif not contexts_placeholders and context:
          # The animal is valid for all contexts
          animal_dict['contexts'].append(context)

    nsamples = num_test_samples // len(animal_placeholders2dict.keys())
    for animal, animal_dict in animal_placeholders2dict.items():
      tests = create_animal_crossproduct(
          editor, animal_dict['contexts'], [q for q in t_entry.questions if q],
          t_entry.answers, animal_dict['animal'],
          [a for a in default_entry.args['vehicle'] if a], nsamples=nsamples)
      if t is None:
        t = tests
      else:
        t += tests
  elif lang in ['ru', 'tr']:
    # Variation in animal and vehicles based on context
    context_placeholders2dict = {}
    for context, en_context in zip([c for c in t_entry.contexts if c],
                                   en_entry.contexts):
      context_placeholders2dict[en_context] = {
          'contexts': [context], 'animal': [], 'vehicle': []}
    for arg_name in ['animal', 'vehicle']:
      for arg_value, arg_placeholders in zip(
          default_entry.args[arg_name],
          default_entry.args[f'{arg_name}_placeholders']):
        for context, context_dict in context_placeholders2dict.items():
          if arg_placeholders and context in arg_placeholders:
            context_dict[arg_name].append(arg_value)
          elif not arg_placeholders and arg_value:
            # The argument is valid for all contexts
            context_dict[arg_name].append(arg_value)

    nsamples = num_test_samples // len(context_placeholders2dict.keys())
    contexts, animals, vehicles = [], set(), set()
    for context, context_dict in context_placeholders2dict.items():
      contexts += context_dict['contexts']
      animals.update(context_dict['animal'])
      vehicles.update(context_dict['vehicle'])
    t = create_animal_crossproduct(
        editor, contexts, [q for q in t_entry.questions if q],
        t_entry.answers, list(animals), list(vehicles), nsamples=nsamples)
  elif lang == 'qu' and name == 'Animal vs Vehicle':
    # Variation in answers based on context
    contexts = [c for c in t_entry.contexts if c]
    nsamples = num_test_samples // 2
    t = create_animal_crossproduct(
        editor, [contexts[0]], t_entry.questions, t_entry.answers,
        [a for a in default_entry.args['animal'] if a],
        [a for a in default_entry.args['vehicle'] if a],
        nsamples=nsamples)
    t += create_animal_crossproduct(
        editor, [contexts[1]], t_entry.questions,
        t_entry.args['answer_placeholders'],
        [a for a in default_entry.args['animal'] if a],
        [a for a in default_entry.args['vehicle'] if a],
        nsamples=nsamples)
  else:
    t = create_animal_crossproduct(
        editor, [c for c in t_entry.contexts if c],
        [q for q in t_entry.questions if q], t_entry.answers,
        [a for a in default_entry.args['animal'] if a],
        [a for a in default_entry.args['vehicle'] if a], num_test_samples)
  return MFT(**t, name=name, expect=Expect.single(
      functools.partial(expect_squad, lang=lang)),
             capability=t_entry.capability)


def create_animal_crossproduct(editor, contexts, questions, answers, animals,
                               vehicles, nsamples=NUM_TEST_SAMPLES):
  """Method for creating tests about animals."""
  return crossproduct(editor.template(
      {
          'contexts': contexts,
          'qas': [(q, a) for q, a in zip(questions, answers)]
      },
      animal=animals,
      vehicle=vehicles,
      remove_duplicates=True,
      nsamples=nsamples,
      save=True,
    ))
