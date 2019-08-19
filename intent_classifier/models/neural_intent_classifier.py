from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("it_classifier")
class IntentClassifier(Model):
    """
    This ``Model`` performs intent classification for user_utterance.  We assume we're given a
    user_utterance, a prev_user_utterance and a prev_sys_utterance, and we predict some output label for intent.
    The basic model structure: we'll embed the user_utterance, the prev_user_utterance and the prev_sys_utterance,
    and encode each of them with separate Seq2VecEncoders, getting a single vector representing the content of each.
    We'll then concatenate those three vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    user_utterance_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the user_utterance to a vector.
    prev_user_utterance_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the prev_user_utterance to a vector.
    prev_sys_utterance_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the prev_sys_utterance to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 user_utterance_encoder: Seq2VecEncoder,
                 prev_user_utterance_encoder: Seq2VecEncoder,
                 prev_sys_utterance_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(IntentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.user_utterance_encoder = user_utterance_encoder
        self.prev_user_utterance_encoder = prev_user_utterance_encoder
        self.prev_sys_utterance_encoder = prev_sys_utterance_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != user_utterance_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the user_utterance_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            user_utterance_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != prev_user_utterance_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the prev_user_utterance_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            prev_user_utterance_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != prev_sys_utterance_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the prev_sys_utterance_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            prev_sys_utterance_encoder.get_input_dim()))

        self.label_accuracy = CategoricalAccuracy()

        self.label_f1_metrics = {}
        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                user_utterance: Dict[str, torch.LongTensor],
                prev_user_utterance: Dict[str, torch.LongTensor],
                prev_sys_utterance: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        user_utterance : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        prev_user_utterance : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        prev_sys_utterance : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the intent label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_user_utterance = self.text_field_embedder(user_utterance)
        user_utterance_mask = util.get_text_field_mask(user_utterance)
        encoded_user_utterance = self.user_utterance_encoder(embedded_user_utterance, user_utterance_mask)

        embedded_prev_user_utterance = self.text_field_embedder(prev_user_utterance)
        prev_user_utterance_mask = util.get_text_field_mask(prev_user_utterance)
        encoded_prev_user_utterance = self.prev_user_utterance_encoder(embedded_prev_user_utterance, prev_user_utterance_mask)

        embedded_prev_sys_utterance = self.text_field_embedder(prev_sys_utterance)
        prev_sys_utterance_mask = util.get_text_field_mask(prev_sys_utterance)
        encoded_prev_sys_utterance = self.prev_sys_utterance_encoder(embedded_prev_sys_utterance, prev_sys_utterance_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_user_utterance, encoded_prev_user_utterance, encoded_prev_sys_utterance], dim=-1))
        class_probs = F.softmax(logits, dim=1)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, label)
            self.label_accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        count = 0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)

            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if metric_val[2]:
                sum_f1 += metric_val[2]
                count += 1

        if count:
            average_f1 = sum_f1 / count
        else:
            average_f1 = sum_f1
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)

        return metric_dict

