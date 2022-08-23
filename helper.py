from torch import nn
from efficientdet_arch.efficientdet.loss import FocalLoss

class ModelWrapper(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss

def freeze_backbone(m, layers=['EfficientNet', 'BiFPN']):
    classname = m.__class__.__name__
    for ntl in layers:
        if ntl in classname:
            for param in m.parameters():
                param.requires_grad = False
                
CONFIG = {"mean": [ 0.485, 0.456, 0.406 ], "std": [ 0.229, 0.224, 0.225 ]}