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

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

from .sb_sed import infer_sentence_breaks

logger = logging.getLogger(__name__)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def retrieval_squad_convert_example_to_features(example,
                                                tokenizer,
                                                max_seq_length,
                                                max_query_length,
                                                max_answer_length):
    _ = max_seq_length  # TODO Currently we don't use max_seq_length, so make sure that it's being used correctly in tokenization.
    features = []
    # TODO make sure that this tokenization is happening correctly with correct cls, sep and max lengths.
    q_features = tokenizer.encode_plus(
        example.question_text,
        max_length=max_query_length,
        pad_to_max_length=True,
    )
    a_features = tokenizer.encode_plus(
        example.sentence_text,
        example.paragraph_text,
        max_length=max_answer_length,
        pad_to_max_length=True
    )

    features.append(
        RetrievalSquadFeatures(
            q_features["input_ids"],
            q_features["attention_mask"],
            q_features["token_type_ids"],
            a_features["input_ids"],
            a_features["attention_mask"],
            a_features["token_type_ids"],
            qas_id=example.qas_id,
            example_index=0,
            unique_id=0,
        )
    )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def retrieval_squad_convert_examples_to_features(
    examples, tokenizer, max_seq_length,
    max_query_length, max_answer_length,
    is_training, return_dataset,
    threads=1,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        max_query_length: The maximum length of the query.
        max_answer_length: The maximum length of the answer and context.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    if return_dataset != 'pt':
        raise NotImplementedError("Retrival Squad can only convert examples to pytorch features.")

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            retrieval_squad_convert_example_to_features,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_query_length=max_query_length,
            max_answer_length=max_answer_length
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
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
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        q_input_ids = torch.tensor([f.q_input_ids for f in features], dtype=torch.long)
        q_attention_masks = torch.tensor([f.q_attention_mask for f in features], dtype=torch.long)
        q_token_type_ids = torch.tensor([f.q_token_type_ids for f in features], dtype=torch.long)
        a_input_ids = torch.tensor([f.a_input_ids for f in features], dtype=torch.long)
        a_attention_masks = torch.tensor([f.a_attention_mask for f in features], dtype=torch.long)
        a_token_type_ids = torch.tensor([f.a_token_type_ids for f in features], dtype=torch.long)
        # TODO more code here based on the model type (similar to logic in squad.py) because input features are different.
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


class RetrievalSquadProcessor(DataProcessor):
    """
    Processor for the Retrieval SQuAD data set.
    """

    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"

    def get_train_examples(self, data_dir, filename=None, language='en'):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train", language)

    def get_dev_examples(self, data_dir, filename=None, language='en'):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev", language)

    def _create_examples(self, input_data, set_type, language):
        is_training = set_type == "train"
        paragraph_id = 0
        examples = []
        for entry in tqdm(input_data):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                sentence_breaks = list(infer_sentence_breaks(paragraph_text))  # TODO can also get sentence_breaks from json directly.
                paragraph_id += 1
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qas in paragraph["qas"]:
                    qas_id = qas["id"]
                    question_text = qas["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    # If a question has multiple answers, we only use the first.
                    answer = qas["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    sentence_text = None
                    for start, end in sentence_breaks:
                        if start <= answer_offset < end:
                            sentence_text = paragraph_text[start:end]
                            break
                    # A potential problem here is that the sentence might break
                    # around the answer fragment. In that case, we skip the example.
                    if not sentence_text:
                        continue
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                        actual_text, cleaned_answer_text)
                        continue

                    example = RetrievalSquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        answer_text=actual_text,
                        sentence_text=sentence_text,
                        paragraph_text=paragraph_text,
                        paragraph_id=paragraph_id)
                    examples.append(example)
        return examples


class RetrievalSquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        answer_text,
        sentence_text,
        paragraph_text,
        paragraph_id
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.answer_text = answer_text
        self.sentence_text = sentence_text
        self.paragraph_text = paragraph_text
        self.paragraph_id = paragraph_id


class RetrievalSquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        q_input_ids,
        q_attention_mask,
        q_token_type_ids,
        a_input_ids,
        a_attention_mask,
        a_token_type_ids,
        qas_id,
        example_index,
        unique_id
    ):
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.q_token_type_ids = q_token_type_ids
        self.a_input_ids = a_input_ids
        self.a_attention_mask = a_attention_mask
        self.a_token_type_ids = a_token_type_ids
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index


class RetrievalSquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, q_encoding, a_encoding):
        self.q_encoding = q_encoding
        self.a_encoding = a_encoding
        self.unique_id = unique_id