from typing import Dict, List, Any
import logging
from overrides import overrides

import torch
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.archival import load_archive


from syntactic_entailment.modules.se_bert_sequence_classification import SyntacticEntailmentBertForSequenceClassification as SEBertForSC


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('se-bert')
class SyntacticEntailmentBert(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 parser_model_path: str,
                 parser_hidden_size: int,
                 parser_cuda_device: int,
                 freeze_parser: bool,
                 pretrained_bert_model_file: str,
                 num_labels: int) -> None:
        super().__init__(vocab)
        self.bert_sc_model = SEBertForSC.from_pretrained(
            pretrained_bert_model_file,
            num_labels=num_labels,
            parser_hidden_size=parser_hidden_size
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._parser = load_archive(parser_model_path,
                                    cuda_device=parser_cuda_device).model
        self._parser._head_sentinel.requires_grad = False
        for child in self._parser.children():
            for param in child.parameters():
                param.requires_grad = False
        if not freeze_parser:
            for param in self._parser.encoder.parameters():
                param.requires_grad = True

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                premise_tags: torch.LongTensor,
                hypothesis: Dict[str, torch.LongTensor],
                hypothesis_tags: torch.LongTensor,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # running the parser
        p_encoded_parse = self._parser(premise, premise_tags)['encoder_final_state']
        h_encoded_parse = self._parser(hypothesis, hypothesis_tags)['encoder_final_state']

        logits = self.bert_sc_model(p_encoded_parse,
                                    h_encoded_parse,
                                    torch.stack(input_ids),
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
