import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.tokenization_bert import whitespace_tokenize
from transformers import DataProcessor

import utils_mewslix

from processors.lareqa import RetrievalSquadFeatures

if is_torch_available():
  import torch
  from torch.utils.data import TensorDataset

if is_tf_available():
  import tensorflow as tf

logger = logging.getLogger(__name__)

MentionEntityPair = utils_mewslix.MentionEntityPair


def wikipedia_el_convert_example_to_features(example, tokenizer,
                                             max_seq_length):
  features = []
  q_features = tokenizer.encode_plus(
      utils_mewslix.preprocess_mention(example.contextual_mention),
      max_length=max_seq_length,
      pad_to_max_length=True,
  )
  a_features = tokenizer.encode_plus(
      utils_mewslix.preprocess_entity_description(example.entity),
      max_length=max_seq_length,
      pad_to_max_length=True)

  ## Helpful for debugging:
  # logger.info("MENTION EXAMPLE")
  # logger.info("%s: %s" % (
  #     example.contextual_mention.mention.mention_span.text,
  #     example.contextual_mention.truncate(0).context.text))
  # logger.info(q_features["input_ids"])
  # logger.info(tokenizer.convert_ids_to_tokens(q_features["input_ids"]))
  # logger.info("ENTITY EXAMPLE")
  # logger.info(example.entity.description)
  # logger.info(a_features["input_ids"])
  # logger.info(tokenizer.convert_ids_to_tokens(a_features["input_ids"]))
  # assert False
  ##

  features.append(
      RetrievalSquadFeatures(
          q_features["input_ids"],
          q_features["attention_mask"],
          q_features["token_type_ids"],
          a_features["input_ids"],
          a_features["attention_mask"],
          a_features["token_type_ids"],
          qas_id=None,
          example_index=0,
          unique_id=0,
      ))
  return features


def wikipedia_el_convert_example_to_features_init(tokenizer_for_convert):
  global tokenizer
  tokenizer = tokenizer_for_convert


def convert_wikipedia_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    is_training,
    return_dataset,
    threads=1,
):
  """Converts a list of examples into a list of features that can be directly given as input to a model.

  It is model-dependant and takes advantage of many of the tokenizer's
  features to create the model's inputs.
  Args:
      examples: list of `MentionEntityPair`
      tokenizer: an instance of a child of
        :class:`~transformers.PreTrainedTokenizer`
      max_seq_length: The maximum sequence length of the inputs.
      is_training: whether to create features for model evaluation or model
        training.
      return_dataset: Default False. Either 'pt' or 'tf'. if 'pt': returns a
        torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
      threads: multiple processing threads

  Returns:
      list of RetrievalSquadFeatures

  Example::

      processor = WikiELProcessor()
      examples = processor.get_dev_examples(data_dir)

      features = convert_wikipedia_examples_to_features(
          examples=examples,
          tokenizer=tokenizer,
          max_seq_length=args.max_seq_length,
          is_training=not evaluate,
      )
  """
  if return_dataset != "pt":
    raise NotImplementedError(
        "WikiEL can only convert examples to pytorch features.")

  # Defining helper methods
  features = []
  threads = min(threads, cpu_count())
  with Pool(
      threads,
      initializer=wikipedia_el_convert_example_to_features_init,
      initargs=(tokenizer,)) as p:
    annotate_ = partial(
        wikipedia_el_convert_example_to_features,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    features = list(
        tqdm(
            p.imap(annotate_, examples, chunksize=32),
            total=len(examples),
            desc="convert wikipedia_el examples to features",
        ))
  new_features = []
  unique_id = 1000000000
  example_index = 0
  for example_features in tqdm(
      features, total=len(features), desc="add example index and unique id"):
    if not example_features:
      continue
    for example_feature in example_features:
      example_feature.example_index = example_index
      example_feature.unique_id = unique_id
      new_features.append(example_feature)
      unique_id += 1
    example_index += 1
  features = new_features
  del new_features
  if return_dataset == "pt":
    if not is_torch_available():
      raise RuntimeError(
          "PyTorch must be installed to return a PyTorch dataset.")

    # Convert to Tensors and build dataset
    q_input_ids = torch.tensor([f.q_input_ids for f in features],
                               dtype=torch.long)
    q_attention_masks = torch.tensor([f.q_attention_mask for f in features],
                                     dtype=torch.long)
    q_token_type_ids = torch.tensor([f.q_token_type_ids for f in features],
                                    dtype=torch.long)
    a_input_ids = torch.tensor([f.a_input_ids for f in features],
                               dtype=torch.long)
    a_attention_masks = torch.tensor([f.a_attention_mask for f in features],
                                     dtype=torch.long)
    a_token_type_ids = torch.tensor([f.a_token_type_ids for f in features],
                                    dtype=torch.long)
    if not is_training:
      all_example_index = torch.arange(q_input_ids.size(0), dtype=torch.long)
      dataset = TensorDataset(
          q_input_ids,
          q_attention_masks,
          q_token_type_ids,
          a_input_ids,
          a_attention_masks,
          a_token_type_ids,
          all_example_index  # Add all_example_index as a feature
      )
    else:
      dataset = TensorDataset(
          q_input_ids,
          q_attention_masks,
          q_token_type_ids,
          a_input_ids,
          a_attention_masks,
          a_token_type_ids,
      )
    return features, dataset
  elif return_dataset == "tf":
    raise NotImplementedError()

  return features


class WikiELProcessor(DataProcessor):
  """Processor for the Wikipedia portion of Mewsli-X entity linking task."""

  train_file = "wikipedia_pairs-train.jsonl"
  dev_file = "wikipedia_pairs-dev.jsonl"

  def get_train_examples(self, data_dir, filename=None):
    """Returns the training examples from the data directory.

    Args:
        data_dir: Directory containing the data files used for training and
          evaluating.
        filename: None by default, specify this if the training file has a
          different name than the original one defined by this class.
    """
    if data_dir is None:
      data_dir = ""

    if self.train_file is None and filename is None:
      raise ValueError("Must specify either train_filename or filename.")

    input_file = os.path.join(data_dir,
                              self.train_file if filename is None else filename)
    return utils_mewslix.load_jsonl(input_file, utils_mewslix.MentionEntityPair)

  def get_dev_examples(self, data_dir, filename=None):
    """Returns the evaluation examples from the data directory.

    Args:
        data_dir: Directory containing the data files used for training and
          evaluating.
        filename: None by default, specify this if the evaluation file has a
          different name than the original one defined by this class.
    """
    if data_dir is None:
      data_dir = ""

    if self.dev_file is None and filename is None:
      raise ValueError("Must specify either dev_file or filename.")

    input_file = os.path.join(data_dir,
                              self.dev_file if filename is None else filename)
    return utils_mewslix.load_jsonl(input_file, utils_mewslix.MentionEntityPair)
