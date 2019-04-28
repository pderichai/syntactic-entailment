import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from allennlp.data.vocabulary import Vocabulary, pop_max_vocab_size
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params, Registrable


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


@Vocabulary.register("se-vocabulary")
class SyntacticEntailmentVocabulary(Vocabulary):

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        parser_vocab = params.pop('parser_vocab')
        pos_vocab = params.pop('pos_vocab')

        vocab_type = params.pop("type", None)
        if vocab_type is not None:
            return cls.by_name(vocab_type).from_params(params=params, instances=instances)

        extend = params.pop("extend", False)
        vocabulary_directory = params.pop("directory_path", None)
        if not vocabulary_directory and not instances:
            raise ConfigurationError("You must provide either a Params object containing a "
                                     "vocab_directory key or a Dataset to build a vocabulary from.")
        if extend and not instances:
            raise ConfigurationError("'extend' is true but there are not instances passed to extend.")
        if extend and not vocabulary_directory:
            raise ConfigurationError("'extend' is true but there is not 'directory_path' to extend from.")

        if vocabulary_directory and instances:
            if extend:
                logger.info("Loading Vocab from files and extending it with dataset.")
            else:
                logger.info("Loading Vocab from files instead of dataset.")

        if vocabulary_directory:
            vocab = cls.from_files(vocabulary_directory)
            if not extend:
                params.assert_empty("Vocabulary - from files")
                return vocab
        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab
        min_count = params.pop("min_count", None)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        min_pretrained_embeddings = params.pop("min_pretrained_embeddings", None)
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)
        params.assert_empty("Vocabulary - from dataset")
        vocab = cls.from_instances(instances=instances,
                                   min_count=min_count,
                                   max_vocab_size=max_vocab_size,
                                   non_padded_namespaces=non_padded_namespaces,
                                   pretrained_files=pretrained_files,
                                   only_include_pretrained_words=only_include_pretrained_words,
                                   tokens_to_add=tokens_to_add,
                                   min_pretrained_embeddings=min_pretrained_embeddings)

        # setting the vocab of the parser from its pretrained files
        vocab.set_from_file(filename=parser_vocab, namespace='tokens')
        vocab.set_from_file(filename=pos_vocab, namespace='pos')

        return vocab
