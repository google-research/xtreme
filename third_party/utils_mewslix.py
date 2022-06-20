"""Utilities for handling Mewsli-X data and evaluation.

This module is adapted from the utilities distributed along with Mewsli-X, i.e.
https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli_x/schema.py
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import json
import pathlib
import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

# Constants related to text preprocessing.
#
# Symbols for marking up a mention span in a text passage.
_MARKER_L = "{"
_MARKER_R = "}"
# Regular expressions for escaping pre-existing instances of the marker symbols.
_REGEX = re.compile(rf"{re.escape(_MARKER_L)}(.*?){re.escape(_MARKER_R)}")
_REPLACEMENT = r"(\1)"

# Constants related to task definition.
#
# Mewsli-X eval languages.
MENTION_LANGUAGES = ("ar", "de", "en", "es", "fa", "ja", "pl", "ro", "ta", "tr",
                     "uk")
# The retrieval task for Mewsli-X in XTREME-R is capped at evaluating the top-K
# retrieved entities.
TOP_K = 20

JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JsonDict = Dict[str, JsonValue]
JsonList = List[JsonValue]
StrOrPurePath = Union[str, pathlib.PurePath]


def to_jsonl(obj: JsonDict) -> str:
  return json.dumps(obj, ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class Span:
  """A [start:end]-span in some external string."""
  start: int
  end: int

  def __post_init__(self):
    if self.start < 0:
      raise ValueError(f"start offset is out of bounds {self}")
    if self.end < 0:
      raise ValueError(f"end offset is out of bounds {self}")
    if self.start >= self.end:
      raise ValueError(f"start and end offsets are non-monotonic {self}")

  @staticmethod
  def from_json(json_dict: JsonDict) -> Span:
    """Creates a new Span instance from the given JSON-dictionary."""
    return Span(start=json_dict["start"], end=json_dict["end"])

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(start=self.start, end=self.end)

  def validate_offsets_relative_to_context(self, context: str) -> None:
    """Validates the span's offsets relative to a context string."""
    if self.start >= len(context):
      raise ValueError(
          f"start offset in {self} is out of bounds w.r.t. '{context}'")
    if self.end > len(context):
      raise ValueError(
          f"end offset in {self} is out of bounds w.r.t. '{context}'")

  def locate_in(self, spans: Iterable[Span]) -> Optional[int]:
    """Returns the index of the first span that fully contains `self`.

    Args:
      spans: The spans to search.

    Returns:
      First i such that spans[i].{start,end} covers `self.{start,end}`, or None
      if there is no such span, indicating that `self` either is out of range
      relative to spans or crosses span boundaries.
    """
    for i, span in enumerate(spans):
      # The starts may coincide and the ends may coincide.
      if (span.start <= self.start and self.start < span.end and
          span.start < self.end and self.end <= span.end):
        return i
    return None


@dataclasses.dataclass(frozen=True)
class TextSpan(Span):
  """A text span relative to an external string T, with text=T[start:end]."""
  text: str

  def validate_relative_to_context(self, context: str) -> None:
    """Validates that `self.text` matches the designated span in `context`."""
    self.validate_offsets_relative_to_context(context)
    ref_text = context[self.start:self.end]
    if self.text != ref_text:
      raise ValueError(f"{self} does not match against context '{context}': "
                       f"'{self.text}' != '{ref_text}'")

  @staticmethod
  def from_context(span: Span, context: str) -> TextSpan:
    """Creates a new TextSpan by extracting the given `span` from `context`."""
    span.validate_offsets_relative_to_context(context)
    return TextSpan(span.start, span.end, text=context[span.start:span.end])

  @staticmethod
  def from_elements(start: int, end: int, context: str) -> TextSpan:
    """Creates a new TextSpan by extracting [start:end] from `context`."""
    return TextSpan.from_context(span=Span(start, end), context=context)

  @staticmethod
  def from_json(json_dict: JsonDict) -> TextSpan:
    """Creates a new TextSpan from the given JSON-dictionary."""
    return TextSpan(
        start=json_dict["start"], end=json_dict["end"], text=json_dict["text"])

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(start=self.start, end=self.end, text=self.text)


@dataclasses.dataclass(frozen=True)
class Entity:
  """An entity and its textual representation.

  Attributes:
    entity_id: Unique identifier of the entity, e.g. WikiData QID.
    title: A title phrase that names the entity.
    description: A definitional description of the entity that serves as its
      unique textual representation, e.g. taken from the beginning of the
      entity's Wikipedia page.
    sentence_spans: Sentence break annotations for the description, as
      character-level Span objects that index into `description`
    sentences: Sentences extracted from `description` according to
      `sentence_spans`. These TextSpan objects include the actual sentence text
      for added convenience. E.g., the string of the description's first
      sentence is `sentences[0].text`.
    description_language: Primary language code of the description and title,
      matching the Wikipedia language edition from which they were extracted.
    description_url: URL of the page where the description was extracted from.
  """
  entity_id: str
  title: str
  description: str
  sentence_spans: Tuple[Span, ...]
  description_language: str
  description_url: str

  def __post_init__(self):
    self.validate()

  @property
  def sentences(self) -> Iterator[TextSpan]:
    for span in self.sentence_spans:
      yield TextSpan.from_context(span, self.description)

  def validate(self):
    for sentence_span in self.sentence_spans:
      sentence_span.validate_offsets_relative_to_context(self.description)

  @staticmethod
  def from_json(json_dict: JsonDict) -> Entity:
    """Creates a new Entity from the given JSON-dictionary."""
    return Entity(
        entity_id=json_dict["entity_id"],
        title=json_dict["title"],
        description=json_dict["description"],
        description_language=json_dict["description_language"],
        description_url=json_dict["description_url"],
        sentence_spans=tuple(
            Span.from_json(t) for t in json_dict["sentence_spans"]),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(
        entity_id=self.entity_id,
        title=self.title,
        description=self.description,
        description_language=self.description_language,
        description_url=self.description_url,
        sentence_spans=[t.to_json() for t in self.sentence_spans],
    )


@dataclasses.dataclass(frozen=True)
class Mention:
  """A single mention of an entity, referring to some external context.

  Attributes:
    example_id: Unique identifier for the mention instance.
    mention_span: A TextSpan denoting one mention, relative to external context.
    entity_id: ID of the mentioned entity.
    metadata: Optional dictionary of additional information about the instance.
  """
  example_id: str
  mention_span: TextSpan
  entity_id: str
  metadata: Optional[Dict[str, str]] = None

  @staticmethod
  def from_json(json_dict: JsonDict) -> Mention:
    """Creates a new Mention from the given JSON-dictionary."""
    return Mention(
        example_id=json_dict["example_id"],
        mention_span=TextSpan.from_json(json_dict["mention_span"]),
        entity_id=json_dict["entity_id"],
        metadata=json_dict.get("metadata"),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        example_id=self.example_id,
        mention_span=self.mention_span.to_json(),
        entity_id=self.entity_id,
    )
    if self.metadata is not None:
      json_dict["metadata"] = self.metadata
    return json_dict


@dataclasses.dataclass()
class Context:
  """A document text fragment and metadata.

  Attributes:
    document_title: Title of the document.
    document_url: URL of the document.
    document_id: An identifier for the document. For a Wikipedia page, this may
      be the associated WikiData QID.
    language: Primary language code of the document.
    text: Original text from the document.
    sentence_spans: Sentence break annotations for the text, as character-level
      Span objects that index into `text`.
    sentences: Sentences extracted from `text` according to `sentence_spans`.
      These TextSpan objects include the actual sentence text for added
      convenience. E.g., the first sentence's string is `sentences[0].text`.
    section_title: Optional title of the section under which `text` appeared.
  """
  document_title: str
  document_url: str
  document_id: str
  language: str
  text: str
  sentence_spans: Tuple[Span, ...]
  section_title: Optional[str] = None

  def __post_init__(self):
    self.validate()

  @property
  def sentences(self) -> Iterator[TextSpan]:
    for span in self.sentence_spans:
      yield TextSpan.from_context(span, self.text)

  def validate(self):
    for sentence_span in self.sentence_spans:
      sentence_span.validate_offsets_relative_to_context(self.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> Context:
    """Creates a new Context from the given JSON-dictionary."""
    return Context(
        document_title=json_dict["document_title"],
        section_title=json_dict.get("section_title"),
        document_url=json_dict["document_url"],
        document_id=json_dict["document_id"],
        language=json_dict["language"],
        text=json_dict["text"],
        sentence_spans=tuple(
            Span.from_json(t) for t in json_dict["sentence_spans"]),
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        document_title=self.document_title,
        document_url=self.document_url,
        document_id=self.document_id,
        language=self.language,
        text=self.text if keep_text else "",
        sentence_spans=[t.to_json() for t in self.sentence_spans],
    )
    if self.section_title is not None:
      json_dict["section_title"] = self.section_title
    return json_dict

  def truncate(self, focus: int, window_size: int) -> Tuple[int, Context]:
    """Truncates the Context to window_size sentences each side of focus.

    This seeks to truncate the text and sentence_spans of `self` to
      self.sentence_spans[focus - window_size:focus + window_size + 1].

    When there are fewer than window_size sentences available before (after) the
    focus, this attempts to retain additional context sentences after (before)
    the focus.

    Args:
      focus: The index of the focus sentence in self.sentence_spans.
      window_size: Number of sentences to retain on each side of the focus.

    Returns:
      - c, the number of characters removed from the start of the text, which is
        useful for updating any Mention defined in relation to this Context.
      - new_context, a copy of the Context that is updated to contain the
        truncated text and sentence_spans.

    Raises:
      IndexError: if focus is not within the range of self.sentence_spans.
      ValueError: if window_size is negative.
    """
    if focus < 0 or focus >= len(self.sentence_spans):
      raise IndexError(f"Index {focus} invalid for {self.sentence_spans}")
    if window_size < 0:
      raise ValueError(f"Expected a positive window, but got {window_size}")

    snt_window = self._get_sentence_window(focus, window_size)
    relevant_sentences = self.sentence_spans[snt_window.start:snt_window.end]

    char_offset = relevant_sentences[0].start
    char_end = relevant_sentences[-1].end
    new_text = self.text[char_offset:char_end]

    new_sentences = [
        Span(old_sentence.start - char_offset, old_sentence.end - char_offset)
        for old_sentence in relevant_sentences
    ]
    new_context = dataclasses.replace(
        self, text=new_text, sentence_spans=tuple(new_sentences))
    return char_offset, new_context

  def _get_sentence_window(self, focus: int, window_size: int) -> Span:
    """Gets Span of sentence indices to cover window around the focus index."""
    # Add window to the left of focus. If there are fewer sentences before the
    # focus sentence, carry over the remainder.
    left_index = max(focus - window_size, 0)
    remainder_left = window_size - (focus - left_index)
    assert remainder_left >= 0, remainder_left

    # Add window to the right of focus, including carryover. (Note, right_index
    # is an inclusive index.) If there are fewer sentences after the focus
    # sentence, carry back the remainder.
    right_index = min(focus + window_size + remainder_left,
                      len(self.sentence_spans) - 1)
    remainder_right = window_size - (right_index - focus)

    if remainder_right > 0:
      # Extend further leftward.
      left_index = max(left_index - remainder_right, 0)

    return Span(left_index, right_index + 1)


@dataclasses.dataclass()
class ContextualMentions:
  """Multiple entity mentions in a shared context."""
  context: Context
  mentions: List[Mention]

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.context.validate()
    for mention in self.mentions:
      mention.mention_span.validate_relative_to_context(self.context.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> ContextualMentions:
    """Creates a new ContextualMentions from the given JSON-dictionary."""
    return ContextualMentions(
        context=Context.from_json(json_dict["context"]),
        mentions=[Mention.from_json(m) for m in json_dict["mentions"]],
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        context=self.context.to_json(keep_text=keep_text),
        mentions=[m.to_json() for m in self.mentions],
    )
    return json_dict

  def unnest_to_single_mention_per_context(self) -> Iterator[ContextualMention]:
    for mention in self.mentions:
      yield ContextualMention(
          context=copy.deepcopy(self.context), mention=copy.deepcopy(mention))

  @staticmethod
  def nest_mentions_by_shared_context(
      contextual_mentions: Iterable[ContextualMention]
  ) -> Iterator[ContextualMentions]:
    """Inverse of unnest_to_single_mention_per_context."""
    contexts = {}
    groups = collections.defaultdict(list)
    for cm in contextual_mentions:
      context = cm.context
      key = (context.document_id, context.section_title, context.text)
      if key in contexts:
        assert contexts[key] == context, key
      else:
        contexts[key] = context
      groups[key].append(cm.mention)

    for key, mentions in groups.items():
      yield ContextualMentions(contexts[key], mentions)


@dataclasses.dataclass()
class ContextualMention:
  """A single entity mention in context."""
  context: Context
  mention: Mention

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.context.validate()
    self.mention.mention_span.validate_relative_to_context(self.context.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> ContextualMention:
    """Creates a new ContextualMention from the given JSON-dictionary."""
    return ContextualMention(
        context=Context.from_json(json_dict["context"]),
        mention=Mention.from_json(json_dict["mention"]),
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        context=self.context.to_json(keep_text=keep_text),
        mention=self.mention.to_json(),
    )
    return json_dict

  def truncate(self, window_size: int) -> Optional[ContextualMention]:
    """Truncates the context to window_size sentences each side of the mention.

    Args:
      window_size: Number of sentences to retain on each side of the sentence
        containing the mention. See Context.truncate for more detail.

    Returns:
      Returns None if no sentence spans were present or if the mention crosses
      sentence boundaries. Otherwise, returns an update copy of the
      ContextualMention where `.context` contains the truncated text and
      sentences, and the character offsets in `.mention` updated accordingly.
    """
    focus_snt = self.mention.mention_span.locate_in(self.context.sentence_spans)
    if focus_snt is None:
      # The context has no sentences or the mention crosses sentence boundaries.
      return None

    offset, new_context = self.context.truncate(
        focus=focus_snt, window_size=window_size)

    # Internal consistency check.
    max_valid = window_size * 2 + 1
    assert len(new_context.sentence_spans) <= max_valid, (
        f"Got {len(new_context.sentence_spans)}>{max_valid} sentences for "
        f"window_size={window_size} in truncated Context: {new_context}")

    new_mention = dataclasses.replace(
        self.mention,
        mention_span=TextSpan(
            start=self.mention.mention_span.start - offset,
            end=self.mention.mention_span.end - offset,
            text=self.mention.mention_span.text))
    return ContextualMention(context=new_context, mention=new_mention)


@dataclasses.dataclass()
class MentionEntityPair:
  """A ContextualMention paired with the Entity it refers to."""
  contextual_mention: ContextualMention
  entity: Entity

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.contextual_mention.validate()
    self.entity.validate()

  @staticmethod
  def from_json(json_dict: JsonDict) -> MentionEntityPair:
    """Creates a new MentionEntityPair from the given JSON-dictionary."""
    return MentionEntityPair(
        contextual_mention=ContextualMention.from_json(
            json_dict["contextual_mention"]),
        entity=Entity.from_json(json_dict["entity"]),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        contextual_mention=self.contextual_mention.to_json(),
        entity=self.entity.to_json(),
    )
    return json_dict


SchemaAnyT = TypeVar("SchemaAnyT", Entity, Context, ContextualMention,
                     ContextualMentions, MentionEntityPair)
SchemaAny = Union[ContextualMention, ContextualMentions, Entity,
                  MentionEntityPair]


def load_jsonl_as_dicts(path: StrOrPurePath) -> List[JsonDict]:
  """Returns dict-records from JSONL file (without parsing into dataclasses)."""
  with open(path) as input_file:
    return [json.loads(line) for line in input_file]


def load_jsonl(path: StrOrPurePath,
               schema_cls: Type[SchemaAnyT]) -> List[SchemaAnyT]:
  """Loads the designated type of schema dataclass items from a JSONL file.

  Args:
    path: File path to load. Each line in the file is a JSON-serialized object.
    schema_cls: The dataclass to parse into, e.g. `ContextualMention`, `Entity`,
      etc.

  Returns:
    A list of validated instances of `schema_cls`, one per input line.
  """
  result = []
  for json_dict in load_jsonl_as_dicts(path):
    result.append(schema_cls.from_json(json_dict))
  return result


def write_jsonl(path: StrOrPurePath, items: Iterable[SchemaAny]) -> None:
  """Writes a list of any of the schema dataclass items to JSONL file.

  Args:
    path: Output file path that will store each item as a JSON-serialized line.
    items: Items to output. Instances of a schema dataclass, e.g.
      `ContextualMention`, `Entity`, etc.
  """
  with open(path, "wt") as output_file:
    for item in items:
      print(to_jsonl(item.to_json()), file=output_file)


# Baseline preprocessing functions. Users are free to create other more nuanced
# approaches.


def preprocess_mention(contextual_mention: ContextualMention) -> str:
  """A baseline method for preprocessing a ContextualMention to string.

  Args:
    contextual_mention: The ContextualMention to preprocess for passing to a
      tokenizer or model.

  Returns:
    A string representing the mention in context. This baseline implementation
    uses one sentence of context, based on sentence annotations provided with
    the Mewsli-X data. The mention span is accentuated by enclosing it in marker
    symbols defined at the top of this module. As a simple scheme to prevent the
    mention span from getting truncated due to a maximum model sequence length,
    it is redundantly prepended. For example:
      Smith: The verdict came back for { Smith }, who stepped down as AG of
      Fictitious County in February, 2012.
  """

  # Take 1 sentence of context text.
  cm = contextual_mention.truncate(window_size=0)
  assert cm is not None, contextual_mention
  left = cm.context.text[:cm.mention.mention_span.start]
  right = cm.context.text[cm.mention.mention_span.end:]

  # Create a context markup that highlights the mention span, while escaping
  # any existing instances of the markers.
  replacements = 0
  left, n = _REGEX.subn(_REPLACEMENT, left)
  replacements += n
  right, n = _REGEX.subn(_REPLACEMENT, right)
  replacements += n

  context_markup = (
      f"{left} {_MARKER_L} {cm.mention.mention_span.text} {_MARKER_R} {right}")

  # Also prepend the mention span to prevent truncation due to limited model
  # sequence lengths.
  query_string = f"{cm.mention.mention_span.text}: {context_markup}"

  # Normalize away newlines and extra spaces.
  query_string = " ".join(query_string.splitlines()).replace("  ", " ").strip()

  if replacements > 0:
    # Escaping does not occur in the WikiNews portion of Mewlis-X, but does
    # occur a handful of times in the Wikipedia data.
    print(f"Applied marker escaping for example_id {cm.mention.example_id}: "
          f"{query_string}")

  return query_string


def preprocess_entity_description(entity: Entity) -> str:
  """Returns a lightly normalized string from an Entity's description."""
  sentences_text = " ".join(s.text for s in entity.sentences)
  # Normalize away newlines and extra spaces.
  return " ".join(sentences_text.splitlines()).replace("  ", " ").strip()


# Evaluation functions.


def mean_reciprocal_rank(golds: Sequence[str],
                         predictions: Sequence[str]) -> float:
  """Computes mean reciprocal rank.

  Args:
    golds: list of gold entity ids.
    predictions: list of prediction lists

  Returns:
    mean reciprocal rank
  """

  def _reciprocal_rank(labels):
    for rank, label in enumerate(labels, start=1):
      if label:
        return 1.0 / rank
    return 0.0

  assert len(golds) == len(predictions)

  reciprocal_ranks = []
  for gold, prediction_list in zip(golds, predictions):
    labels = [int(p == gold) for p in prediction_list]
    reciprocal_ranks.append(_reciprocal_rank(labels))
  if reciprocal_ranks:
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
  return 0.0


def evaluate(
    gold_data: Mapping[str, Sequence[str]],
    predictions: Mapping[str, Sequence[str]],
    k: int = TOP_K,
    return_count_diff: bool = False) -> Union[float, Tuple[float, int, int]]:
  """Evaluates one set of entity linking predictions.

  Args:
    gold_data: dictionary that maps each unique example_id to a *singleton list*
      containing its gold entity ID.
    predictions: dictionary that maps each unique example_id to a list of
      predicted entity IDs. Partial credit may be obtained even if some
      instances are missing from this dictionary.
    k: Number of top predictions to evaluate per evaluation instance.
    return_count_diff: Whether to also return the number of missing and
      unexpected example_ids in predictions.

  Returns:
    mean reciprocal rank in the interval [0, 1] by default, otherwise if
    return_count_diff is True, returns the tuple
     (mean reciprocal rank, num missing example_ids, num extra example_ids).
  """
  # The dataset has a single gold label per instance, but the file provides
  # it as a list.
  gold_single_labels = {
      ex_id: labels[0].strip() for ex_id, labels in gold_data.items()
  }

  # Convert to parallel lists, and truncate to top-k predictions.
  gold_ids: List[str] = []
  pred_ids: List[List[str]] = []
  for example_id, gold in gold_single_labels.items():
    top_k_preds = predictions.get(example_id, [])[:k]
    gold_ids.append(gold)
    pred_ids.append(top_k_preds)
  assert len(gold_ids) == len(pred_ids), (len(gold_ids), len(pred_ids))

  mrr = mean_reciprocal_rank(golds=gold_ids, predictions=pred_ids)
  if return_count_diff:
    unpredicted_count = len(set(gold_single_labels) - set(predictions))
    unexpected_count = len(set(predictions) - set(gold_single_labels))
    return mrr, unpredicted_count, unexpected_count
  else:
    return mrr


def evaluate_all(gold_file_pattern: str,
                 pred_file_pattern: str,
                 output_path: Optional[str] = None,
                 k: int = TOP_K) -> float:
  """Evaluates entity linking predictions from per-language files.

  Args:
    gold_file_pattern: file pattern for per-language gold labels, specified as
      an f-string with argument "lang" for language, e.g. "gold-{lang}.json",
      containing a single dictionary that maps each unique example id to a
      *singleton list* with its gold entity ID. Assumes that files exist for all
      languages in MENTION_LANGUAGES.
    pred_file_pattern: file pattern for per-language predictions, specified as
      an f-string with argument "lang" for language, e.g. "preds-{lang}.json",
      containing a single dictionary that maps each unique example id to a list
      of predicted entity IDs. Scoring proceeds even if some predictions are
      missing, in which case an alert is printed.
    output_path: Path to write results to. If None, results are printed to
      standard output.
    k: Number of top predictions to evaluate.

  Returns:
    mean reciprocal rank in interval [0, 1], macro-averaged over languages.
  """
  outputs = []
  columns = ["", f"MRR/MAP@{k}"]
  outputs.append("\t".join(columns))

  unpredicted_count = 0
  unexpected_count = 0

  mrr_sum = 0.0
  for language in sorted(MENTION_LANGUAGES):
    # Read the data for the language into dictionaries keyed on example_id,
    # which allows evaluating incomplete predictions.
    with open(gold_file_pattern.format(lang=language)) as f:
      gold_data: Dict[str, List[str]] = json.load(f)

    predictions: Dict[str, List[str]] = {}
    pred_path = pathlib.Path(pred_file_pattern.format(lang=language))
    if pred_path.exists():
      with open(pred_path) as f:
        predictions = json.load(f)

    mrr, unpredicted, unexpected = evaluate(
        gold_data=gold_data,
        predictions=predictions,
        k=k,
        return_count_diff=True)
    mrr_sum += mrr
    columns = [language, f"{mrr * 100:.2f}"]
    outputs.append("\t".join(columns))

    unpredicted_count += unpredicted
    unexpected_count += unexpected

  macro_avg = mrr_sum / len(MENTION_LANGUAGES)
  columns = ["MACRO_AVG", f"{macro_avg * 100:.2f}"]
  outputs.append("\t".join(columns))

  report_text = "\n".join(outputs)
  if output_path is None:
    print(report_text)
  else:
    with open(output_path, "w") as f:
      f.write(report_text + "\n")

  if unpredicted_count:
    print(f"Gold examples without predictions: {unpredicted_count}")
  if unexpected_count:
    print(f"Predicted examples without gold: {unexpected_count}")

  return macro_avg
