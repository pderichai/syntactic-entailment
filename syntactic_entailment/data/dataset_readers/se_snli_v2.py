from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("se-snli-v6")
class SyntacticEntailmentSnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        premise_tags = [x.tag_ for x in premise_tokens]
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        hypothesis_tags = [x.tag_ for x in hypothesis_tokens]

        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        fields['premise_tags'] = SequenceLabelField(premise_tags,
                                                    fields['premise'],
                                                    label_namespace="pos")
        fields['hypothesis_tags'] = SequenceLabelField(hypothesis_tags,
                                                       fields['hypothesis'],
                                                       label_namespace="pos")
        if label:
            fields['label'] = LabelField(label)

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                    "premise_tags": [x.tag_ for x in premise_tokens],
                    "hypothesis_tags": [x.tag_ for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
