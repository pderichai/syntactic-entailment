from typing import Dict, Any, List, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

@Predictor.register('syntactic-entailment-dependency-parser')
class SyntacticEntailmentDependencyParserPredictor(BiaffineDependencyParserPredictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        # spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        # sentence_text = [token.text for token in spacy_tokens]
        # if self._dataset_reader.use_language_specific_pos: # type: ignore
        #     # fine-grained part of speech
        #     pos_tags = [token.tag_ for token in spacy_tokens]
        # else:
        #     # coarse-grained part of speech (Universal Depdendencies format)
        #     pos_tags = [token.pos_ for token in spacy_tokens]
        sentence_text = json_dict["sentence"]
        pos_tags = json_dict["tags"]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)
