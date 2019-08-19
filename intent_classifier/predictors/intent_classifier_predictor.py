from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('it_classifier')
class IntentClassifierPredictor(Predictor):
    """"Predictor wrapper for the IntentClassifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        # label_dict will be like {0: "positive_preference", 1: "no_intent", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["positive_preference", "no_intent", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_utterance = json_dict['user_utterance']
        prev_user_utterance = json_dict['prev_user_utterance']
        prev_sys_utterance = json_dict['prev_sys_utterance']
        return self._dataset_reader.text_to_instance(user_utterance=user_utterance, prev_user_utterance=prev_user_utterance, prev_sys_utterance=prev_sys_utterance)