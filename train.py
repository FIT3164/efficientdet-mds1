import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from efficientdet_arch.efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet_arch.backbone import EfficientDetBackbone
from efficientdet_arch.efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet_arch.utils.utils import preprocess, postprocess, invert_affine, display
from helper import *  # including config

with open('train_config.json') as f:
    config = json.load(f)

INPUT_DIM = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

PROJECT_NAME = config["PROJECT_NAME"]
EFFICIENTNET_COMPOUND_COEF = config["EFFICIENTNET_COMPOUND_COEF"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCH_NUM = config["EPOCH_NUM"]
LEARNING_RATE = config["LEARNING_RATE"]
OPTIMIZER = config["OPTIMIZER"]
WEIGHT_PATH = config["WEIGHT_PATH"]
ANCHOR_RATIOS = [eval(tup) for tup in config["ANCHOR_RATIOS"]]
ANCHOR_SCALES = [eval(expr) for expr in config["ANCHOR_SCALES"]]

DATASET_DIR = os.path.join(os.path.abspath(os.getcwd()), "datasets", PROJECT_NAME)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    CocoDataset(
        root_dir=DATASET_DIR, set="train",
        transform=transforms.Compose([
            Normalizer(mean=CONFIG["mean"], std=CONFIG["std"]),
            Augmenter(),
            Resizer(INPUT_DIM[EFFICIENTNET_COMPOUND_COEF])
        ])
    ), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collater
)

model = EfficientDetBackbone(num_classes=2, compound_coef=0, ratios=ANCHOR_RATIOS, scales=ANCHOR_SCALES)
try:
    missing_keys, unexpected_keys = model.load_state_dict(torch.load("efficientdet-d0.pth"), strict=False)
except Exception as e:
    print(e, "(Omit)")

model.apply(freeze_backbone)
model = ModelWrapper(model, debug=False)
model.to(DEVICE)
model.train()

if OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
elif OPTIMIZER == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.9, nesterov=True)
else:
    raise Exception("Wrong Optimizer Option")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

for epoch in range(EPOCH_NUM):
    epoch_loss = []
    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):
        try:
            imgs, annot = data['img'], data['annot']
            imgs, annot = imgs.to(DEVICE), annot.to(DEVICE)

            optimizer.zero_grad()
            cls_loss, reg_loss = model(imgs, annot, obj_list=["good_chili", "bad_chili"])
            cls_loss, reg_loss = cls_loss.mean(), reg_loss.mean()
            loss = cls_loss + reg_loss

            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            progress_bar.set_description(
                f"""Epoch: {epoch}/{EPOCH_NUM} | Iteration: {i+1}/{len(train_loader)} | Cls loss: {cls_loss.item():.5f} | Reg loss: {reg_loss.item():.5f} | Total loss: {loss.item():.5f}"""
            )
        except ValueError as e:
            print(f"[Error] {e}")
    scheduler.step(np.mean(epoch_loss))

torch.save(model.model.state_dict(), WEIGHT_PATH)