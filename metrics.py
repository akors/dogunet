
from typing import List, Dict

import torch

import monai.metrics



class MultiMetrics():
    metric_names = [
        "precision",
        "recall",
        "accuracy",
        "f1 score", # aka Dice Score
        "threat score" # aka Intersection over Union aka Jaccard Index
    ]
    def __init__(self) -> None:
        self.confm_metric = monai.metrics.ConfusionMatrixMetric(metric_name=MultiMetrics.metric_names)
    
    def update(self, pred, target):
        with torch.no_grad():
            mask_onehot = torch.zeros_like(pred).scatter_(1, target.to(dtype=torch.long), 1.)
    
            pred_amax = torch.argmax(pred, dim=1, keepdim=True).to(dtype=torch.long)
            pred_binarized = torch.zeros_like(pred).scatter_(1, pred_amax, 1.)
        
            m = self.confm_metric(y_pred=pred_binarized, y=mask_onehot)
        return m

    def calculate(self) -> Dict[str, float]:
        results = dict()
        
        # calculate aggregate per-class metrics, will sum up all pixels that were added with
        # the update() method.
        #
        # We can then average over the individual classs
        aggregate_metrics = self.confm_metric.aggregate(reduction="sum_batch")

        perclass_metric = aggregate_metrics[0]
        results['MeanPrecision'] = perclass_metric.nanmean().item()

        perclass_metric = aggregate_metrics[1]
        results['MeanRecall'] = perclass_metric.nanmean().item()

        perclass_metric = aggregate_metrics[2]
        results['MeanAccuracy'] = perclass_metric.nanmean().item()

        perclass_metric = aggregate_metrics[3]
        results['MeanDice'] = perclass_metric.nanmean().item()

        perclass_metric = aggregate_metrics[4]
        results['MeanIoU'] = perclass_metric.nanmean().item()

        confm = self.confm_metric.get_buffer()
        tp = confm[:,:,0]
        fp = confm[:,:,1]

        # ConfusionMatrixMetric aggregates hits per-class. We can recover the overall hits by summing over only the true
        # positives and NOT the true negatives, because the true negatives from each class will be contained within the
        # true positives of the matching class
        results['OverallAccuracy'] = (tp.sum() / (tp+fp).sum()).item()

        return results

    def reset(self):
        self.confm_metric.reset()
