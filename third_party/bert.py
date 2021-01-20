"""BERT dual encoder model for retrieval."""

import torch
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForRetrieval(BertPreTrainedModel):
  """BERT dual encoder model for retrieval."""

  def __init__(self, config, model_attr_name='bert', model_cls=BertModel):
    super().__init__(config)

    self.model_attr_name = model_attr_name
    self.model_cls = model_cls

    # Set model attribute, e.g. self.bert = BertModel(config)
    setattr(self, model_attr_name, model_cls(config))

    def normalized_cls_token(cls_token):
      return torch.nn.functional.normalize(cls_token, p=2, dim=1)
    self.normalized_cls_token = normalized_cls_token
    self.logit_scale = torch.nn.Parameter(torch.empty(1))
    torch.nn.init.constant_(self.logit_scale, 100.0)
    self.init_weights()

  def model(self):
    return getattr(self, self.model_attr_name)

  def forward(
      self,
      q_input_ids=None,
      q_attention_mask=None,
      q_token_type_ids=None,
      a_input_ids=None,
      a_attention_mask=None,
      a_token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      inference=False):
    outputs_a = self.model()(
        q_input_ids,
        attention_mask=q_attention_mask,
        token_type_ids=q_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds)
    if inference:
      # In inference mode, only use the first tower to get the encodings.
      return self.normalized_cls_token(outputs_a[1])

    outputs_b = self.model()(
        a_input_ids,
        attention_mask=a_attention_mask,
        token_type_ids=a_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds)

    a_encodings = self.normalized_cls_token(outputs_a[1])
    b_encodings = self.normalized_cls_token(outputs_b[1])
    similarity = torch.matmul(a_encodings, torch.transpose(b_encodings, 0, 1))
    logits = similarity * self.logit_scale
    batch_size = list(a_encodings.size())[0]
    labels = torch.arange(0, batch_size, device=logits.device)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    return loss, a_encodings, b_encodings
