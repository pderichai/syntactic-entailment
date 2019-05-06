from typing import Dict, List, Any
import logging

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertForSequenceClassification as HuggingFaceBertSC

from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('bert-nli')
class BertForSequenceClassification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_bert_model_file: str,
                 num_labels: int) -> None:
        super().__init__(vocab)
        self.bert_sc_model = HuggingFaceBertSC.from_pretrained(
            pretrained_bert_model_file, num_labels=num_labels)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    @overrides
    def forward(self,  # type: ignore
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        logits = self.bert_sc_model(torch.stack(input_ids),
                                    torch.stack(token_type_ids),
                                    torch.stack(attention_mask))
        output_dict = {"logits": logits}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }
