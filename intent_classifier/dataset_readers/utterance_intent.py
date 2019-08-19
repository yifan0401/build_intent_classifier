from typing import Dict, List, Union
import logging
import jsonlines

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("it_utterance")
class IntentClassificationDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing user_utterance, prev_user_utterance, prev_user_utterance, class(intents),
    annotations, and creates a dataset suitable for intent classification using these utterances.
    Expected format for each input line: {"class":"text","annotations":{"intent":"text",
    "params":[{"start": number, "end": number, "param": "text"}]},"prev_sys_utterance":"text",
    "prev_user_utterance":"text","user_utterance":"text"}

    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        user_utterance: ``TextField``
        prev_user_utterance: ``TextField``
        pre_sys_utterance:``TextField``
        label: ``LabelField``
    where the ``intent`` is derived from the intent of the utterance.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the user_utterance, prev_user_utterance and prev_sys_utterance into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r+", encoding="utf8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in jsonlines.Reader(data_file):
                user_utterance = line['user_utterance']
                prev_user_utterance = line['prev_user_utterance']
                prev_sys_utterance = line['prev_sys_utterance']
                intent = line['class']
                yield self.text_to_instance(user_utterance, prev_user_utterance, prev_sys_utterance, intent)

    @overrides
    def text_to_instance(self, user_utterance: str, prev_user_utterance: str, prev_sys_utterance: str, intent: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_user_utterance = self._tokenizer.tokenize(user_utterance)
        tokenized_prev_user_utterance = self._tokenizer.tokenize(prev_user_utterance)
        tokenized_prev_sys_utterance = self._tokenizer.tokenize(prev_sys_utterance)
        user_utterance_field = TextField(tokenized_user_utterance, self._token_indexers)
        prev_user_utterance_field = TextField(tokenized_prev_user_utterance, self._token_indexers)
        prev_sys_utterance_field = TextField(tokenized_prev_sys_utterance, self._token_indexers)
        fields = {'user_utterance': user_utterance_field,
                  'prev_user_utterance': prev_user_utterance_field,
                  'prev_sys_utterance': prev_sys_utterance_field}
        if intent is not None:
            fields['label'] = LabelField(intent, label_namespace="labels")
        return Instance(fields)


