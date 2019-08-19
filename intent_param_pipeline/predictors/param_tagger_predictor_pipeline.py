from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('pm_tagger_pipeline_predictor')
class ParamTaggerPipelinePredictor(Predictor):
    """
    Predictor for any model that takes in a user_utterance and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)
        # solve the kernel size greater than input size error
        self._dataset_reader._token_indexers['token_characters']._min_padding_length = 5

    def predict(self, user_utterance: str) -> JsonDict:
        return self.predict_json({"user_utterance" : user_utterance})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"user_utterance": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        user_utterance = json_dict["user_utterance"]
        intent = json_dict["class"]
        tokens = self._tokenizer.split_words(user_utterance)
        return self._dataset_reader.text_to_instance(intent,tokens)
