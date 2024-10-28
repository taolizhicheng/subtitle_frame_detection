import numpy as np
from practices.metric.base.base_metric import BaseMetric

from .. import METRIC_BUILDER


@METRIC_BUILDER.register("SubtitleFrameDetectionMetric")
class SubtitleFrameDetectionMetric(BaseMetric):
    def __init__(
        self, 
        num_classes: int = 5,
    ):
        super().__init__(
            num_classes = num_classes,
        )
        self.num_classes = num_classes

    def init(self):
        self.preds = []
        self.labels = []
        self.probs = []

    def reset(self):
        self.preds.clear()
        self.labels.clear()
        self.probs.clear()

    def update(self, preds, labels):
        for (code, probs), label in zip(preds, labels):
            self.preds.append(code)
            self.labels.append(label)
            self.probs.append(probs)

    def compute(self):
        accuracy = self._get_accuracy()
        precision = self._get_precision()
        recall = self._get_recall()
        f1_score = self._get_f1_score()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def _get_accuracy(self):
        return (np.array(self.preds) == np.array(self.labels)).mean()
    
    def _get_precision(self):
        precisions = []
        weights = []
        for i in range(self.num_classes):
            tp = np.sum((np.array(self.preds) == i) & (np.array(self.labels) == i))
            fp = np.sum((np.array(self.preds) == i) & (np.array(self.labels) != i))
            precision = tp / (tp + fp + 1e-6)
            precisions.append(precision)

            num_i = np.sum(np.array(self.labels) == i)
            weights.append(num_i)
        
        return np.average(precisions, weights=weights)

    def _get_recall(self):
        recalls = []
        weights = []
        for i in range(self.num_classes):
            tp = np.sum((np.array(self.preds) == i) & (np.array(self.labels) == i))
            fn = np.sum((np.array(self.preds) != i) & (np.array(self.labels) == i))
            recall = tp / (tp + fn + 1e-6)
            recalls.append(recall)

            num_i = np.sum(np.array(self.labels) == i)
            weights.append(num_i)
        
        return np.average(recalls, weights=weights)

    def _get_f1_score(self):
        precision = self._get_precision()
        recall = self._get_recall()
        return 2 * precision * recall / (precision + recall + 1e-6)
