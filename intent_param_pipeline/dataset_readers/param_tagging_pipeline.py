from typing import Dict, List
import logging
import jsonlines
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "###"

@DatasetReader.register("pm_pipeline")
class ParamTaggingPipelineDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n
    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.
    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
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
        # if `file_path` is a URL, redirect to the cache
        with open(cached_path(file_path), "r+", encoding="utf8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in jsonlines.Reader(data_file):
                # read intent labels
                intent = line['class']

                # read param tags
                user_utterance = line['user_utterance']
                line = line['annotations']
                params = line['params']

                # skip blank lines
                if params:
                    # turn user utterance into list
                    user_utterance_split = re.findall(r"[\w'-]+|[^\s\w,:*!]", user_utterance)
                    for param in params:
                        # fine value of param
                        param = param['param']

                        # turn user param into list
                        param_split = re.findall(r"[\w'-]+|[^\s\w,:*!]", param)

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
                yield self.text_to_instance(intent, tokens, tags)

    @overrides
    def text_to_instance(self, intent: str, tokens: List[Token], tags: List[str] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        tokenized_intent = self._tokenizer.tokenize(intent)
        fields['label'] = TextField(tokenized_intent, self._token_indexers)

        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence, label_namespace="tags")
        return Instance(fields)




