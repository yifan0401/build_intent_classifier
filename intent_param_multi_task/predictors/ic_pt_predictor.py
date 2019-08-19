from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('it_pm_predictor')
class IntentParamPredictor(Predictor):
    """"Predictor wrapper for the IntentClassifier"""
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)
        # solve the kernel size greater than input size error
        #self._dataset_reader._token_indexers['token_characters']._min_padding_length = 5

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        # label_dict will be like {0: "positive_preference", 1: "no_intent", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["positive_preference", "no_intent", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        output_dict["user_utterance"] = inputs['user_utterance']
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_utterance = json_dict['user_utterance']
        prev_user_utterance = json_dict['prev_user_utterance']
        prev_sys_utterance = json_dict['prev_sys_utterance']
        tokens = self._tokenizer.split_words(user_utterance)
        intent = json_dict['class']
        return self._dataset_reader.text_to_instance(user_utterance=user_utterance, prev_user_utterance=prev_user_utterance, prev_sys_utterance=prev_sys_utterance, tokens=tokens, intent=intent)

