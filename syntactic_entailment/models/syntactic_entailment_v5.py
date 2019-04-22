from typing import Dict, Optional, List, Any

import torch

from allennlp.predictors.predictor import Predictor
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("syntactic-entailment-v5")
class SyntacticEntailment(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 project_syntax: FeedForward,
                 encode_syntax: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 parser_model_path: str,
                 predictor_name: str,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SyntacticEntailment, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._project_syntax = TimeDistributed(project_syntax)
        self._attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               attend_feedforward.get_input_dim(),
                               "text field embedding dim",
                               "attend feedforward input dim")
        check_dimensions_match(aggregate_feedforward.get_output_dim(),
                               self._num_labels,
                               "final output dimension",
                               "number of labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._predictor = Predictor.from_path(parser_model_path,
                predictor_name=predictor_name)

        self._device = torch.device("cuda:0" if torch.cuda.is_available()
                                             else "cpu")
        self._predictor._model = self._predictor._model.to(self._device)

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        p_tokens = [metadata[idx]['premise_tokens'] for idx in range(len(metadata))]
        p_tags = [metadata[idx]['premise_tags'] for idx in range(len(metadata))]
        p_jsons = [{'sentence' : p_tokens[idx], 'tags' : p_tags[idx]} for idx in range(len(metadata))]
        h_tokens = [metadata[idx]['hypothesis_tokens'] for idx in range(len(metadata))]
        h_tags = [metadata[idx]['hypothesis_tags'] for idx in range(len(metadata))]
        h_jsons = [{'sentence' : h_tokens[idx], 'tags' : h_tags[idx]} for idx in range(len(metadata))]
        p_encoded_parse = torch.tensor(
                [output['encoded_text']
                        for output in self._predictor.predict_batch_json(p_jsons)]
                ).to(self._device)
        h_encoded_parse = torch.tensor(
                [output['encoded_text']
                        for output in self._predictor.predict_batch_json(h_jsons)]
                ).to(self._device)

        #print()
        #print('p_encoded_parse:', p_encoded_parse[0][0])
        #exit(1)

        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        projected_p_encoded_parse = self._project_syntax(p_encoded_parse)
        projected_h_encoded_parse = self._project_syntax(h_encoded_parse)
        encoded_p_and_syntax = torch.cat((projected_premise, projected_p_encoded_parse), 2)
        encoded_h_and_syntax = torch.cat((projected_hypothesis, projected_h_encoded_parse), 2)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._attention(encoded_p_and_syntax,
                                            encoded_h_and_syntax)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs,
                       "h2p_attention": h2p_attention,
                       "p2h_attention": p2h_attention}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }
