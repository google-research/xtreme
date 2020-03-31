from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_xlm import XLMModel, XLMPreTrainedModel, XLM_START_DOCSTRING, XLM_INPUTS_DOCSTRING
from transformers import XLMConfig, add_start_docstrings

@add_start_docstrings("""XLM Model with a token classification head on top (a linear layer on top of
            the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
            XLM_START_DOCSTRING,
            XLM_INPUTS_DOCSTRING)
class XLMForTokenClassification(XLMPreTrainedModel):
  r"""
    **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
      Labels for computing the token classification loss.
      Indices should be in ``[0, ..., config.num_labels - 1]``.
  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
    **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
      Classification loss.
    **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
      Classification scores (before SoftMax).
    **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
      list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
      of shape ``(batch_size, sequence_length, hidden_size)``:
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    **attentions**: (`optional`, returned when ``config.output_attentions=True``)
      list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
  Examples::
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    model = XLMForTokenClassification.from_pretrained('xlm-mlm-100-1280')
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=labels)
    loss, scores = outputs[:2]
  """
  def __init__(self, config):
    super(XLMForTokenClassification, self).__init__(config)
    self.num_labels = config.num_labels
    self.transformer = XLMModel(config)
    self.dropout = nn.Dropout(config.dropout)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    self.init_weights()
    

  def forward(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

    outputs = self.transformer(input_ids,
              attention_mask=attention_mask,
              langs=langs,
              token_type_ids=token_type_ids,
              position_ids=position_ids,
              head_mask=head_mask,
              inputs_embeds=inputs_embeds)

    sequence_output = outputs[0]

    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
      else:
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), scores, (hidden_states), (attentions)
