import sys,os 
os.environ['DETECTRON2_DATASETS']='datasets'
from detectron2.utils.logger import create_small_table
import numpy as np
import itertools
from tabulate import tabulate
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager



def _derive_coco_results(precisions, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        # metrics = {
        #     "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        #     "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        #     "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        # }[iou_type]
        # # the standard metrics
        results = {}

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        #precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            print(precisions.shape)
            precision = precisions[:, :, idx, 0]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results,table


