from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


@Predictor.register('syntactic-entailment-constituency-parser')
class SyntacticEntailmentConstituencyParserPredictor(ConstituencyParserPredictor):
    """
    Predictor for the :class:`~allennlp.models.SpanConstituencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a constituency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the constituency tree.
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        pos_tags = [token.tag_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)
