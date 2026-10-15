"""Boundary metrics using the project's per-image, GT-present-class averaging.

Unlike dataset-accumulated Boundary IoU protocols, this implementation explicitly
mirrors utils.train_and_eval.mean_iou's class selection and averaging. Ignore IDs
are not independently removed from predicted masks or their unions.
"""
import numpy as np
from scipy import ndimage as ndi


def inner_band(mask, radius):
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), bool),
                               iterations=radius, border_value=0)
    return mask & ~eroded


def dilate(mask, radius):
    return ndi.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=radius)


def contact_pixels(target, valid):
    """Both endpoints of a 4-neighbor edge between distinct foreground classes."""
    out = np.zeros(target.shape, bool)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        edge = valid[a] & valid[b] & (target[a] > 0) & (target[b] > 0)
        edge &= target[a] != target[b]
        out[a] |= edge
        out[b] |= edge
    return out


class BoundaryMetrics:
    def __init__(self, num_classes, boundary_ratio=0.02, tolerance_ratio=0.005):
        if num_classes < 2 or boundary_ratio <= 0 or tolerance_ratio <= 0:
            raise ValueError('Invalid class count or boundary ratios')
        self.classes = num_classes
        self.boundary_ratio, self.tolerance_ratio = boundary_ratio, tolerance_ratio
        self.rows = []
        self.class_sums = np.zeros((num_classes, 2), dtype=np.float64)
        self.class_counts = np.zeros(num_classes, dtype=np.int64)

    def update(self, prediction, target):
        """Score ONE image, like mean_iou with batch_size=1. Return its scores."""
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError('Prediction and target must be equal-shaped 2D class-ID arrays')
        if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(target.dtype, np.integer):
            raise ValueError('Expected integer class IDs, not probability or RGB masks')
        if np.any((prediction < 0) | (prediction >= self.classes)):
            raise ValueError('Prediction contains out-of-range class IDs')
        diagonal = np.hypot(*target.shape)
        d = max(1, int(round(self.boundary_ratio * diagonal)))
        t = max(1, int(round(self.tolerance_ratio * diagonal)))
        valid = (target >= 0) & (target < self.classes)
        # This ROI is a separate diagnostic; no global confusion matrix is used.
        contact = dilate(contact_pixels(target, valid), d) & valid & (target > 0)
        bious, bf1s, cious = [], [], []
        for c in range(self.classes):
            gm = target == c
            if not gm.any():  # Exactly the legacy GT-present-class selection.
                continue
            pm = prediction == c
            pb, gb = inner_band(pm, d), inner_band(gm, d)
            union = np.count_nonzero(pb | gb)
            biou = np.count_nonzero(pb & gb) / union if union else 0.0
            pe, ge = inner_band(pm, 1), inner_band(gm, 1)
            pc, gc = np.count_nonzero(pe), np.count_nonzero(ge)
            precision = np.count_nonzero(pe & dilate(ge, t)) / pc if pc else 0.0
            recall = np.count_nonzero(ge & dilate(pe, t)) / gc if gc else 0.0
            bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            bious.append(biou)
            bf1s.append(bf1)
            self.class_sums[c] += [biou, bf1]
            self.class_counts[c] += 1
            if (gm & contact).any():
                cu = np.count_nonzero((pm | gm) & contact)
                cious.append(np.count_nonzero(pm & gm & contact) / cu if cu else 0.0)
        # Empty class lists return 0, just as the project's mean_iou does.
        average = lambda values: float(np.mean(values)) if values else 0.0
        row = {'boundary_iou': average(bious), 'boundary_f1': average(bf1s),
               'contact_iou': average(cious), 'contact_pixels': int(contact.sum())}
        self.rows.append(row)
        return row

    def compute(self):
        if not self.rows:
            raise ValueError('No images evaluated')
        return {**{key: float(np.mean([r[key] for r in self.rows]))
                   for key in ('boundary_iou', 'boundary_f1', 'contact_iou')},
                'images': len(self.rows),
                'contact_images': sum(r['contact_pixels'] > 0 for r in self.rows),
                'contact_pixels': sum(r['contact_pixels'] for r in self.rows),
                'definition': {'aggregation': 'GT-present classes per image, then equal mean over images',
                               'include_background_if_present': True,
                               'absent_gt_class': 'skip, even when predicted',
                               'ignore': 'legacy semantics: no extra exclusion from prediction union',
                               'boundary_ratio': self.boundary_ratio,
                               'tolerance_ratio': self.tolerance_ratio,
                               'distance': 'square morphology (Chebyshev)',
                               'empty_class_list': 0.0, 'empty_contact_image': 0.0},
                'per_class': [{'class_id': c, 'gt_present_images': int(self.class_counts[c]),
                               'boundary_iou': float(self.class_sums[c, 0] / self.class_counts[c])
                                               if self.class_counts[c] else None,
                               'boundary_f1': float(self.class_sums[c, 1] / self.class_counts[c])
                                              if self.class_counts[c] else None}
                              for c in range(self.classes)]}
