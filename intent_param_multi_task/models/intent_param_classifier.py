from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure


@Model.register("it_pm_classifier")
class IntentParamClassifier(Model):
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
                 encoder: Seq2SeqEncoder,
                 calculate_span_f1: bool = None,
                 tag_encoding: Optional[str] = None,
                 tag_namespace: str = "tags",
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(IntentParamClassifier, self).__init__(vocab, regularizer)

        # Intent task
        self.text_field_embedder = text_field_embedder
        self.label_num_classes = self.vocab.get_vocab_size("labels")
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
        for i in range(self.label_num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()


        # Param task
        self.tag_namespace = tag_namespace
        self.tag_num_classes = self.vocab.get_vocab_size(tag_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.tag_num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        # We keep calculate_span_f1 as a constructor argument for API consistency with
        # the CrfTagger, even it is redundant in this class
        # (tag_encoding serves the same purpose).
        if calculate_span_f1 and not tag_encoding:
            raise ConfigurationError("calculate_span_f1 is True, but "
                                     "no tag_encoding was specified.")

        self.tag_accuracy = CategoricalAccuracy()

        if calculate_span_f1 or tag_encoding:
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=tag_namespace,
                                                 tag_encoding=tag_encoding)
        else:
            self._f1_metric = None

        self.f1 = SpanBasedF1Measure(vocab, tag_namespace=tag_namespace)

        self.tag_f1_metrics = {}
        for k in range(self.tag_num_classes):
            self.tag_f1_metrics[vocab.get_token_from_index(index=k, namespace=tag_namespace)] = F1Measure(
                positive_label=k)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                user_utterance: Dict[str, torch.LongTensor],
                prev_user_utterance: Dict[str, torch.LongTensor],
                prev_sys_utterance: Dict[str, torch.LongTensor],
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
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

        # Intent task
        embedded_user_utterance = self.text_field_embedder(user_utterance)
        user_utterance_mask = util.get_text_field_mask(user_utterance)
        encoded_user_utterance = self.user_utterance_encoder(embedded_user_utterance, user_utterance_mask)

        embedded_prev_user_utterance = self.text_field_embedder(prev_user_utterance)
        prev_user_utterance_mask = util.get_text_field_mask(prev_user_utterance)
        encoded_prev_user_utterance = self.prev_user_utterance_encoder(embedded_prev_user_utterance, prev_user_utterance_mask)

        embedded_prev_sys_utterance = self.text_field_embedder(prev_sys_utterance)
        prev_sys_utterance_mask = util.get_text_field_mask(prev_sys_utterance)
        encoded_prev_sys_utterance = self.prev_sys_utterance_encoder(embedded_prev_sys_utterance, prev_sys_utterance_mask)

        # Param task
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        label_logits = self.classifier_feedforward(torch.cat([encoded_user_utterance, encoded_prev_user_utterance, encoded_prev_sys_utterance], dim=-1))
        label_class_probs = F.softmax(label_logits, dim=1)
        output_dict = {"label_logits": label_logits, "label_class_probs": label_class_probs}

        tag_logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = tag_logits.view(-1, self.tag_num_classes)
        tag_class_probs = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.tag_num_classes])

        output_dict["tag_logits"] = tag_logits
        output_dict["tag_class_probs"] = tag_class_probs

        if label is not None:
            if tags is not None:
                loss = self.loss(label_logits, label) + sequence_cross_entropy_with_logits(tag_logits, tags, mask)
                output_dict["loss"] = loss

                # compute intent F1 per label
                for i in range(self.label_num_classes):
                    metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                    metric(label_class_probs, label)
                self.label_accuracy(label_logits, label)

                # compute param F1 per tag
                for i in range(self.tag_num_classes):
                    metric = self.tag_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="tags")]
                    metric(tag_class_probs, tags, mask.float())
                self.tag_accuracy(tag_logits, tags, mask.float())

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        # Intent task
#        label_class_probs = F.softmax(output_dict["label_logits"], dim=-1)
#        output_dict["label_class_probs"] = label_class_probs
        label_class_probs = output_dict["label_class_probs"]

        label_predictions = label_class_probs.cpu().data.numpy()
        label_argmax_indices = numpy.argmax(label_predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in label_argmax_indices]
        output_dict["label"] = labels

        # Param task
        tag_all_predictions = output_dict["tag_class_probs"]
        tag_all_predictions = tag_all_predictions.cpu().data.numpy()
        if tag_all_predictions.ndim == 3:
            tag_predictions_list = [tag_all_predictions[i] for i in range(tag_all_predictions.shape[0])]
        else:
            tag_predictions_list = [tag_all_predictions]
        all_tags = []
        for tag_predictions in tag_predictions_list:
            tag_argmax_indices = numpy.argmax(tag_predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(y, namespace="tags")
                    for y in tag_argmax_indices]
            all_tags.append(tags)
        output_dict["tags"] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        # intent task
        label_sum_f1 = 0.0
        label_count = 0
        for label_name, label_metric in self.label_f1_metrics.items():
            label_metric_val = label_metric.get_metric(reset)

            metric_dict[label_name + "_P"] = label_metric_val[0]
            metric_dict[label_name + "_R"] = label_metric_val[1]
            metric_dict[label_name + "_F1"] = label_metric_val[2]
            if label_metric_val[2]:
                label_sum_f1 += label_metric_val[2]
                label_count += 1

        if label_count:
            label_average_f1 = label_sum_f1 / label_count
        else:
            label_average_f1 = label_sum_f1
        metric_dict["intent_average_F1"] = label_average_f1
        metric_dict["label_accuracy"] = self.label_accuracy.get_metric(reset)

        # param task
        tag_sum_f1 = 0.0
        tag_count = 0
        for tag_name, tag_metric in self.tag_f1_metrics.items():
            tag_metric_val = tag_metric.get_metric(reset)
            # if self.verbose_metrics:
            metric_dict[tag_name + "_P"] = tag_metric_val[0]
            metric_dict[tag_name + "_R"] = tag_metric_val[1]
            metric_dict[tag_name + "_F1"] = tag_metric_val[2]
            if tag_metric_val[2]:
                tag_sum_f1 += tag_metric_val[2]
                tag_count += 1

        if tag_count:
            tag_average_f1 = tag_sum_f1 / tag_count
        else:
            tag_average_f1 = tag_sum_f1
        metric_dict["param_average_F1"] = tag_average_f1
        metric_dict["tag_accuracy"] = self.tag_accuracy.get_metric(reset)

        return metric_dict
