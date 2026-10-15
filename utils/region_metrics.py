"""Additional diagnostics: GT-present classes per image, then equal image mean."""
import numpy as np


def per_class_regions(pred, target, classes):
    rows = []
    for c in range(classes):
        gt, pr = target == c, pred == c
        n = int(gt.sum())
        if not n:
            continue
        p, tp = int(pr.sum()), int((gt & pr).sum())
        rows.append({'class_id': c, 'gt_pixels': n, 'pred_pixels': p,
                     'iou': tp / (n + p - tp), 'dice': 2 * tp / (n + p),
                     'precision': tp / p if p else 0.0, 'recall': tp / n})
    return rows


def region_summary(records, classes):
    keys = ('iou', 'dice', 'precision', 'recall')
    per_class = []
    for c in range(classes):
        rows = [r for image in records for r in image['per_class_regions'] if r['class_id'] == c]
        per_class.append({'class_id': c, 'gt_present_images': len(rows),
                          **{k: float(np.mean([r[k] for r in rows])) if rows else None for k in keys}})
    return {'protocol': 'Each metric: mean over GT-present classes per image, then equal image mean; per-class values average only GT-present images. Not global confusion matrix.',
            'per_class': per_class,
            **{k: float(np.mean([np.mean([r[k] for r in image['per_class_regions']])
                                if image['per_class_regions'] else 0.0 for image in records])) for k in keys}}
