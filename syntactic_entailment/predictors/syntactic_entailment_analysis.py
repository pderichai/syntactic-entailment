from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from numpy import argmax


@Predictor.register('se-analysis')
class SyntacticEntailmentAnalysisPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.
        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.
        hypothesis : ``str``
            A sentence that may be entailed by the premise.
        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise" : premise, "hypothesis": hypothesis})

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # Some models only have 'logits' key.
        # label_idx = argmax(outputs['logits'])
        label_idx = argmax(outputs['label_probs'])
        return (self._model.vocab.get_token_from_index(label_idx, namespace='labels') + '\t'
                + outputs['gold_label'] + '\t'
                + outputs['premise'] + '\t'
                + outputs['hypothesis'] + '\n')

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        #output_dict['pair_id'] = inputs['pairID']
        output_dict['gold_label'] = inputs['gold_label']
        output_dict['premise'] = inputs['sentence1']
        output_dict['hypothesis'] = inputs['sentence2']
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        # mnli format
        premise_text = json_dict['sentence1']
        hypothesis_text = json_dict['sentence2']

        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)
