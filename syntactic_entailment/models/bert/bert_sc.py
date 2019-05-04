from typing import Dict, List, Any
import collections
import logging
import math

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertForSequenceClassification as HuggingFaceBertSC
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.tokenization import BasicTokenizer

from allennlp.common import JsonDict
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }


@Model.register('bert-sc')
class BertForSequenceClassification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model_type: str,
                 pretrained_archive_path: str,
                 null_score_difference_threshold: float,
                 n_best_size: int = 20,
                 max_answer_length: int = 30) -> None:
        super().__init__(vocab)
        #if bert_model_type == "bert_base":
        #    config_to_use = BERT_BASE_CONFIG
        #elif bert_model_type == "bert_large":
        #    config_to_use = BERT_LARGE_CONFIG
        #else:
        #    raise RuntimeError(f"`bert_model_type` should either be \"bert_large\" or \"bert_base\"")
        #config = BertConfig(vocab_size_or_config_json_file=config_to_use["vocab_size"],
        #                    hidden_size=config_to_use["hidden_size"],
        #                    num_hidden_layers=config_to_use["num_hidden_layers"],
        #                    num_attention_heads=config_to_use["num_attention_heads"],
        #                    intermediate_size=config_to_use["intermediate_size"],
        #                    hidden_act=config_to_use["hidden_act"],
        #                    hidden_dropout_prob=config_to_use["hidden_dropout_prob"],
        #                    attention_probs_dropout_prob=config_to_use["attention_probs_dropout_prob"],
        #                    max_position_embeddings=config_to_use["max_position_embeddings"],
        #                    type_vocab_size=config_to_use["type_vocab_size"],
        #                    initializer_range=config_to_use["initializer_range"])
        self.bert_sc_model = HuggingFaceBertSC.from_pretrained(bert_model_type, num_labels=2)
        self._loaded_sc_weights = False
        #self._pretrained_archive_path = pretrained_archive_path
        self._null_score_difference_threshold = null_score_difference_threshold
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    @overrides
    def forward(self,  # type: ignore
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                #premise: Dict[str, torch.LongTensor],
                #premise_tags,
                #hypothesis: Dict[str, torch.LongTensor],
                #hypothesis_tags,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        #if not self._loaded_sc_weights and self.training:
        #    self.bert_sc_model = HuggingFaceBertSC.from_pretrained(self._pretrained_archive_path, num_labels=2)
        #    self._loaded_sc_weights = True
        assert self.bert_sc_model is not None
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
