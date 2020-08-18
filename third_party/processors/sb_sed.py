"""A sed-style sentence breaker, with SQuAD/Wiki-specific tweaks.

This code is derived from a broad-coverage proof-of-concept sed script,
with some simplification/adaptation:

  - Several substitution rules are simplified here to focus on the cases that
    arise in the SQuAD sentences.
  - Other substitution rules are added to deal with specifics of the SQuAD
    corpus.

This code aims for high accuracy/F1, but does not try to reach full 100%.
Some of the long tail is left alone, to avoid proliferating substitution rules
for diminishing returns.

When run in squad_sentence_break_test.py, sb_sed achieves an F1 of .9961.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re


def infer_sentence_breaks(uni_text):
  """Generates (start, end) pairs demarking sentences in the text.

  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.

  Yields:
    (start, end) tuples that demarcate sentences in the input text. Normal
    Python slicing applies: the start index points at the first character of
    the sentence, and the end index is one past the last character of the
    sentence.
  """
  # Treat the text as a single line that starts out with no internal newline
  # characters and, after regexp-governed substitutions, contains internal
  # newlines representing cuts between sentences.
  uni_text = re.sub(r'\n', r' ', uni_text)  # Remove pre-existing newlines.
  text_with_breaks = _sed_do_sentence_breaks(uni_text)
  starts = [m.end() for m in re.finditer(r'^\s*', text_with_breaks, re.M)]
  sentences = [s.strip() for s in text_with_breaks.split('\n')]
  assert len(starts) == len(sentences)
  for i in range(len(sentences)):
    start = starts[i]
    end = start + len(sentences[i])
    yield start, end


def _sed_do_sentence_breaks(uni_text):
  """Uses regexp substitution rules to insert newlines as sentence breaks.

  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.

  Returns:
    A Unicode string with internal newlines representing the inferred sentence
    breaks.
  """

  # The main split, looks for sequence of:
  #   - sentence-ending punctuation: [.?!]
  #   - optional quotes, parens, spaces: [)'" \u201D]*
  #   - whitespace: \s
  #   - optional whitespace: \s*
  #   - optional opening quotes, bracket, paren: [['"(\u201C]?
  #   - upper case letter or digit
  txt = re.sub(
      r'''([.?!][)'" %s]*)\s(\s*[['"(%s]?[A-Z0-9])''' % ('\u201D', '\u201C'),
      r'\1\n\2', uni_text)

  # Wiki-specific split, for sentence-final editorial scraps (which can stack):
  #  - ".[citation needed]", ".[note 1] ", ".[c] ", ".[n 8] "
  txt = re.sub(r'''([.?!]['"]?)((\[[a-zA-Z0-9 ?]+\])+)\s(\s*['"(]?[A-Z0-9])''',
               r'\1\2\n\4', txt)

  # Wiki-specific split, for ellipses in multi-sentence quotes:
  # "need such things [...] But"
  txt = re.sub(r'(\[\.\.\.\]\s*)\s(\[?[A-Z])', r'\1\n\2', txt)

  # Rejoin for:
  #   - social, military, religious, and professional titles
  #   - common literary abbreviations
  #   - month name abbreviations
  #   - geographical abbreviations
  #
  txt = re.sub(r'\b(Mrs?|Ms|Dr|Prof|Fr|Rev|Msgr|Sta?)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Lt|Gen|Col|Maj|Adm|Capt|Sgt|Rep|Gov|Sen|Pres)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(e\.g|i\.?e|vs?|pp?|cf|a\.k\.a|approx|app|es[pt]|tr)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(Jan|Aug|Oct|Nov|Dec)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Mt|Ft)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b([ap]\.m)\.\n(Eastern|EST)\b', r'\1. \2', txt)

  # Rejoin for personal names with 3,2, or 1 initials preceding the last name.
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3 \4',
               txt)
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3',
               txt)
  txt = re.sub(r'\b([A-Z]\.[A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)
  txt = re.sub(r'\b([A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)

  # Resplit for common sentence starts:
  #   - The, This, That, ...
  #   - Meanwhile, However,
  #   - In, On, By, During, After, ...
  txt = re.sub(r'([.!?][\'")]*) (The|This|That|These|It) ', r'\1\n\2 ', txt)
  txt = re.sub(r'(\.) (Meanwhile|However)', r'\1\n\2', txt)
  txt = re.sub(r'(\.) (In|On|By|During|After|Under|Although|Yet|As |Several'
               r'|According to) ',
               r'\1\n\2 ',
               txt)

  # Rejoin for:
  #   - numbered parts of documents.
  #   - born, died, ruled, circa, flourished ...
  #   - et al (2005), ...
  #   - H.R. 2000
  txt = re.sub(r'\b([Aa]rt|[Nn]o|Opp?|ch|Sec|cl|Rec|Ecl|Cor|Lk|Jn|Vol)\.\n'
               r'([0-9IVX]+)\b',
               r'\1. \2',
               txt)
  txt = re.sub(r'\b([bdrc]|ca|fl)\.\n([A-Z0-9])', r'\1. \2', txt)
  txt = re.sub(r'\b(et al)\.\n(\(?[0-9]{4}\b)', r'\1. \2', txt)
  txt = re.sub(r'\b(H\.R\.)\n([0-9])', r'\1 \2', txt)

  # SQuAD-specific joins.
  txt = re.sub(r'(I Am\.\.\.)\n(Sasha Fierce|World Tour)', r'\1 \2', txt)
  txt = re.sub(r'(Warner Bros\.)\n(Records|Entertainment)', r'\1 \2', txt)
  txt = re.sub(r'(U\.S\.)\n(\(?\d\d+)', r'\1 \2', txt)
  txt = re.sub(r'\b(Rs\.)\n(\d)', r'\1 \2', txt)

  # SQuAD-specific splits.
  txt = re.sub(r'\b(Jay Z\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Washington, D\.C\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(for 4\.\)) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Wii U\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\. (iPod|iTunes)', r'.\n\1', txt)
  txt = re.sub(r' (\[\.\.\.\]\n)', r'\n\1', txt)
  txt = re.sub(r'(\.Sc\.)\n', r'\1 ', txt)
  txt = re.sub(r' (%s [A-Z])' % '\u2022', r'\n\1', txt)
  return txt