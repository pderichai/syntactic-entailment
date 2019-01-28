from typing import Dict, Tuple, List, Optional, NamedTuple, Any
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.util import get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer, DEFAULT_EVALB_DIR
from allennlp.common.checks import ConfigurationError

from allennlp.models.constituency_parser import SpanConstituencyParser
from allennlp.models.constituency_parser import SpanInformation

@Model.register("syntactic_entailment_constituency_parser")
class SyntacticEntailmentSpanConstituencyParser(SpanConstituencyParser):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    encoder : ``Seq2SeqEncoder``, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward : ``FeedForward``, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    evalb_directory_path : ``str``, optional (default=``DEFAULT_EVALB_DIR``)
        The path to the directory containing the EVALB executable used to score
        bracketed parses. By default, will use the EVALB included with allennlp,
        which is located at allennlp/tools/EVALB . If ``None``, EVALB scoring
        is not used.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor,
                 encoder: Seq2SeqEncoder,
                 feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 evalb_directory_path: str = DEFAULT_EVALB_DIR) -> None:
        super(SyntacticEntailmentSpanConstituencyParser, self).__init__(vocab,
                                                                        text_field_embedder,
                                                                        span_extractor,
                                                                        encoder,
                                                                        feedforward,
                                                                        pos_tag_embedding,
                                                                        initializer,
                                                                        regularizer,
                                                                        evalb_directory_path)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                pos_tags: Dict[str, torch.LongTensor] = None,
                span_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        spans : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)`` representing the
            inclusive start and end indices of all possible spans in the sentence.
        metadata : List[Dict[str, Any]], required.
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
                gold_tree : ``nltk.Tree``, optional (default = None)
                    Gold NLTK trees for use in evaluation.
                pos_tags : ``List[str]``, optional.
                    The POS tags for the sentence. These can be used in the
                    model as embedded features, but they are passed here
                    in addition for use in constructing the tree.
        pos_tags : ``torch.LongTensor``, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        span_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        spans : ``torch.LongTensor``
            The original spans tensor.
        tokens : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, required.
            A list of POS tags in the sentence for each element in the batch.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self.pos_tag_embedding is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        if span_mask.dim() == 1:
            # This happens if you use batch_size 1 and encounter
            # a length 1 sentence in PTB, which do exist. -.-
            span_mask = span_mask.unsqueeze(-1)
        if span_labels is not None and span_labels.dim() == 1:
            span_labels = span_labels.unsqueeze(-1)

        num_spans = get_lengths_from_binary_sequence_mask(span_mask)

        encoded_text = self.encoder(embedded_text_input, mask)
        #encoder_final_state = get_final_encoder_states(encoded_text, mask)

        #output_dict = {
        #        "encoder_final_state": encoder_final_state
        #}

        output_dict = {
                "encoded_text": encoded_text
        }

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs an NLTK ``Tree`` given the scored spans. We also switch to exclusive
        span ends when constructing the tree representation, because it makes indexing
        into lists cleaner for ranges of text, rather than individual indices.

        Finally, for batch prediction, we will have padded spans and class probabilities.
        In order to make this less confusing, we remove all the padded spans and
        distributions from ``spans`` and ``class_probabilities`` respectively.
        """
        return output_dict
