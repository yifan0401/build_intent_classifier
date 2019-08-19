from typing import Dict, List, Union
import logging
# import json
import jsonlines
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "###"

@DatasetReader.register("it_pm")
class IntentParamDatasetReader(DatasetReader):
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
                 word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r+", encoding="utf8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in jsonlines.Reader(data_file):
                user_utterance = line['user_utterance']
                prev_user_utterance = line['prev_user_utterance']
                prev_sys_utterance = line['prev_sys_utterance']
                intent = line['class']

                # user_utterance = line['user_utterance']
                line = line['annotations']
                params = line['params']
                # param = line['param']
                # not_param_part = user_utterance.replace(param, "").strip()

                # skip blank lines
                if params:
                    # turn user utterance into list
                    user_utterance_split = re.findall(r"[\w'-]+|[^\s\w,:*!]", user_utterance)
                    for param in params:
                        # fine value of param
                        param = param['param']

                        # turn user param into list
                        param_split = re.findall(r"[\w'-]+|[^\s\w,:*!]", param)
                        # user_utterance_split = user_utterance.split()
                        # param_split = param.split()

                        # lower the letters in user_utterance and parameters
                        user_utterance_split_lower = [x.lower() for x in user_utterance_split]
                        param_split_lower = [x.lower() for x in param_split]

                        if param_split_lower[0] not in user_utterance_split_lower:
                            for item in user_utterance_split_lower:
                                if param_split_lower[0] in item:
                                    param_split_lower[0] = item

                        # get index of param words start and end indexes
                        param_start = user_utterance_split_lower.index(param_split_lower[0])
                        param_end = param_start + len(param_split_lower)

                        # add "###B_Param" tag to param first word
                        user_utterance_split[param_start] += "###B_Param "
                        param_I_start = param_start + 1

                        # add "###I_Param" tag to param rest word
                        for i in range(param_I_start, param_end):
                            user_utterance_split[i] += "###I_Param "

                    # add "###O" to each rest word (non-param) of user utterance
                    for k in range(len(user_utterance_split)):
                        if "###" not in user_utterance_split[k]:
                            user_utterance_split[k] += "###O "

                    user_utterance_with_tagger = ''.join(user_utterance_split).strip()

                else:
                    user_utterance_split = re.findall(r"[\w'-]+|[^\s\w,:*!]", user_utterance)
                    for k in range(len(user_utterance_split)):
                        user_utterance_split[k] += "###O "

                    user_utterance_with_tagger = ''.join(user_utterance_split).strip()

                tokens_and_tags = [pair.rsplit(self._word_tag_delimiter, 1)
                                   for pair in user_utterance_with_tagger.split(self._token_delimiter)]
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]

                yield self.text_to_instance(user_utterance, prev_user_utterance, prev_sys_utterance, intent, tokens, tags)

    @overrides
    def text_to_instance(self, user_utterance: str, prev_user_utterance: str, prev_sys_utterance: str, intent: str, tokens: List[Token], tags: List[str] = None) -> Instance:  # type: ignore
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

        # fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence, label_namespace="tags")

        return Instance(fields)

