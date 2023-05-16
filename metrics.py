
from typing import Any, List, Dict, Tuple
import numpy as np

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
        self.overall_correct_pixels = 0
        self.overall_pixels = 0
    
    def calculate_overall_accuracy(self) -> float:
        return float(self.overall_correct_pixels) / float(self.overall_pixels)

    def update(self, pred, target, detailed=True):
        with torch.no_grad():
            pred_amax = torch.argmax(pred, dim=1, keepdim=True).to(dtype=torch.long)
            
            self.overall_correct_pixels += (pred_amax == target).sum().item()
            self.overall_pixels +=  target.shape[0] * target.shape[-2] * target.shape[-1]

            if not detailed:
                return None
            
            mask_onehot = torch.zeros_like(pred).scatter_(1, target.to(dtype=torch.long), 1.)
            pred_binarized = torch.zeros_like(pred).scatter_(1, pred_amax, 1.)
            m = self.confm_metric(y_pred=pred_binarized, y=mask_onehot)

        return m

    def calculate(self, detailed=True) -> Dict[str, float]:
        results = dict()

        results['OverallAccuracy'] = self.calculate_overall_accuracy()

        if not detailed:
            return results

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

        return results


    def reset(self):
        self.confm_metric.reset()
        self.overall_correct_pixels = 0
        self.overall_pixels = 0


class DatasetMetricsFirstpass():
    def __init__(self):
        self.num_samples = 0

        # mean of RGB values over whole dataset
        self.rgb_means: List[List] = []
        self.heights: List[float] = []
        self.widths: List[float] = []

        self.class_pixelcount: List[Dict[int, int]] = []

    def update(self, img, mask):
        shape = img.shape

        # get mean of RGB channels
        rgb_mean = img.mean(dim=(-2,-1))
        self.rgb_means.append(rgb_mean.tolist())

        self.heights.append(shape[-2])
        self.widths.append(shape[-1])

        unique_classes, unique_classes_count = mask.unique(return_counts=True)
        self.class_pixelcount.append({
            i.item() : c.item() for i, c in zip(unique_classes, unique_classes_count)
        })

        self.num_samples += 1
        pass
    
    def concatenate_with(self, other):
        self.num_samples += other.num_samples

        # append per batch stat list of other firstpass stat lists
        self.rgb_means.extend(other.rgb_means)
        self.heights.extend(other.heights)
        self.widths.extend(other.widths)
        self.class_pixelcount.extend(other.class_pixelcount)

        pass


    def calculate(self):
        result = dict()
        
        result['sample_count'] = self.num_samples

        a = np.array(self.rgb_means)
        result['rgb_mean'] = np.mean(a, axis=0).tolist()

        a = np.array(self.heights)
        result['height_mean'] = np.mean(a)
        result['height_std'] = np.std(a)

        a = np.array(self.widths)
        result['width_mean'] = np.mean(a)
        result['width_std'] = np.std(a)

        class_pixelcounts: Dict[int, int] = dict()
        class_objectcounts: Dict[int, int] = dict()
        for imgclasses in self.class_pixelcount:
            for classidx, pixelcount_of_class in imgclasses.items():
                if classidx not in class_pixelcounts.keys():
                    class_pixelcounts[classidx] = 0
                    class_objectcounts[classidx] = 0

                class_pixelcounts[classidx] += pixelcount_of_class
                class_objectcounts[classidx] += (pixelcount_of_class > 0)

        #result['numclasses_mean'] = self.numclasses_mean

        # sort pixel,object counts by label
        class_pixelcounts = dict(sorted(class_pixelcounts.items()))
        result['class_pixels'] = class_pixelcounts

        class_objectcounts = dict(sorted(class_objectcounts.items()))
        result['class_objects'] = class_objectcounts

        # total pixel count
        result['pixel_count'] = 0
        for p in class_pixelcounts.values():
            result['pixel_count'] += p

        # total object count
        result['object_count'] = 0
        for p in class_objectcounts.values():
            result['object_count'] += p

        return result

class DatasetMetricsSecondpass():
    # standard deviation for peasants

    def __init__(self, firstpass_results: Dict[str, Any]):
        self.num_total_samples = firstpass_results['sample_count']
        self.total_pixels = firstpass_results['pixel_count']

        # mean values from first pass
        self.rgb_global_mean_t = torch.tensor(firstpass_results['rgb_mean']).unsqueeze(-1).unsqueeze(-1)

        # mean of RGB values over whole dataset
        self.rgb_squaresums = np.empty((self.num_total_samples, 3), dtype=np.double)
        self.num_pixels = np.empty(self.num_total_samples, dtype=np.uint)

    def update(self, img: torch.Tensor, mask, sample_idx):
        self.num_pixels[sample_idx] = (img.shape[-2] * img.shape[-1])

        # subtract global rgb mean from values, then calculate the sum of the squares
        rgb_squaresum = (img - self.rgb_global_mean_t).square().sum(dim=(-2,-1))
        self.rgb_squaresums[sample_idx,:] = rgb_squaresum.numpy()

        pass

    def calculate(self):
        result = dict()

        variance = np.sum(self.rgb_squaresums, axis=0) / (self.total_pixels - 1)
        std = np.sqrt(variance)
        result['rgb_std'] = [c for c in std]

        return result

