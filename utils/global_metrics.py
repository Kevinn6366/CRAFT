"""Dataset-accumulated confusion matrix: rows GT, columns prediction."""
import numpy as np
import torch

class GlobalIoU:
    def __init__(self, classes):
        self.classes = classes
        self.matrix = np.zeros((classes,classes),dtype=np.int64)
        self.images = 0

    def update(self, prediction, target):
        p,g = np.asarray(prediction),np.asarray(target)
        if p.shape != g.shape:
            raise ValueError('Prediction/GT shape mismatch')
        valid = (g>=0)&(g<self.classes)
        if np.any((p[valid]<0)|(p[valid]>=self.classes)):
            raise ValueError('Invalid prediction class')
        self.matrix += np.bincount((g[valid].astype(np.int64)*self.classes+p[valid]).ravel(),
                                  minlength=self.classes**2).reshape(self.matrix.shape)
        self.images += 1

    def compute(self):
        cm = self.matrix
        tp=np.diag(cm); union=cm.sum(0)+cm.sum(1)-tp
        iou=np.divide(tp,union,out=np.zeros(self.classes,dtype=float),where=union>0)
        valid=union>0
        fg=valid.copy(); fg[0]=False
        return {'miou':float(iou[valid].mean()) if valid.any() else 0.,
                'miou_without_background':float(iou[fg].mean()) if fg.any() else 0.,
                'images':self.images,'valid_pixels':int(cm.sum()),
                'per_class_iou':[float(iou[c]) if valid[c] else None for c in range(self.classes)],
                'confusion_matrix':cm.tolist(),'definition':{'aggregation':'dataset accumulated confusion matrix, macro IoU over union>0 classes',
                'include_background':True,'ignore':'GT outside [0,num_classes) excluded','units':'fractions'}}

@torch.no_grad()
def evaluate_global(model, loader, device, classes):
    model.eval()
    metric=GlobalIoU(classes)
    for images,target,_ in loader:
        pred=model(images.to(device))[0].argmax(1).cpu().numpy()
        for p,g in zip(pred,target.numpy()): metric.update(p,g)
    r=metric.compute()
    return {'Mean IoU':r['miou'],'Mean Accuracy':r['miou'],
            'selection_alias':'Mean Accuracy key contains global mIoU for checkpoint selection',
            'miou_without_background':r['miou_without_background'],'images':r['images']}
