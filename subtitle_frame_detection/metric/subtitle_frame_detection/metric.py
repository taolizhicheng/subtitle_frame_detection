from practices.metric.base.base_metric import BaseMetric

from .. import METRIC_BUILDER


@METRIC_BUILDER.register("SubtitleFrameDetectionMetric")
class SubtitleFrameDetectionMetric(BaseMetric):
    def __init__(
        self, 
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        super().__init__(
            score_threshold = score_threshold,
            iou_threshold = iou_threshold,
        )
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        
        self.reset()

    def reset(self):
        self.preds = []
        self.labels = []
        self.ious = []
        self.results = {
            "labels": {
                "summary": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                },
                "details": {
                    "tp": [],
                    "fp": [],
                    "fn": [],
                    "tn": [],
                },
            },
            "ious": {
                "summary": {
                    "matched": 0,
                    "unmatched": 0,
                },
                "details": {
                    "matched": [],
                    "unmatched": [],
                },
            },
        }

    def update(self, preds, labels):
        index = len(self.preds)
        for pred, label in zip(preds, labels):
            pred_rect = pred[:4]
            pred_score = pred[4]
            label_rect = label[:4]
            label_score = label[4]

            iou = self._iou(pred_rect, label_rect)
            self.ious.append(iou)

            if label_score == 1:
                if pred_score > self.score_threshold:
                    self.results["labels"]["summary"]["tp"] += 1
                    self.results["labels"]["details"]["tp"].append(index)
                else:
                    self.results["labels"]["summary"]["fn"] += 1
                    self.results["labels"]["details"]["fn"].append(index)
            else:
                if pred_score > self.score_threshold:
                    self.results["labels"]["summary"]["fp"] += 1
                    self.results["labels"]["details"]["fp"].append(index)
                else:
                    self.results["labels"]["summary"]["tn"] += 1
                    self.results["labels"]["details"]["tn"].append(index)
            
            if iou > self.iou_threshold:
                self.results["ious"]["summary"]["matched"] += 1
                self.results["ious"]["details"]["matched"].append(index)
            else:
                self.results["ious"]["summary"]["unmatched"] += 1
                self.results["ious"]["details"]["unmatched"].append(index)
            
            index += 1

        self.preds.extend(preds)
        self.labels.extend(labels)

    def compute(self):
        tp = self.results["labels"]["summary"]["tp"]
        fp = self.results["labels"]["summary"]["fp"]
        fn = self.results["labels"]["summary"]["fn"]
        tn = self.results["labels"]["summary"]["tn"]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        iou_matched = self.results["ious"]["summary"]["matched"]
        iou_unmatched = self.results["ious"]["summary"]["unmatched"]
        iou_match_rate = iou_matched / (iou_matched + iou_unmatched)
        
        return {
            "precision": precision,
            "recall": recall,
            "iou_match_rate": iou_match_rate,
        }

    def _iou(self, pred_rect, label_rect):
        cx1, cy1, w1, h1 = pred_rect
        cx2, cy2, w2, h2 = label_rect

        if (cx1 < 0 or cy1 < 0 or w1 < 0 or h1 < 0) and (cx2 < 0 or cy2 < 0 or w2 < 0 or h2 < 0):
            return 1

        x1_max = cx1 + w1 / 2
        x1_min = cx1 - w1 / 2
        y1_max = cy1 + h1 / 2
        y1_min = cy1 - h1 / 2

        x2_max = cx2 + w2 / 2
        x2_min = cx2 - w2 / 2
        y2_max = cy2 + h2 / 2
        y2_min = cy2 - h2 / 2

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        overlap_area = x_overlap * y_overlap
        pred_area = w1 * h1
        label_area = w2 * h2

        iou = overlap_area / (pred_area + label_area - overlap_area)
        return iou
