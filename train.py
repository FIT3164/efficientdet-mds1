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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from helper import *  # including config

with open('train_config.json') as f:
    config = json.load(f)

INPUT_DIM = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

PROJECT_NAME = config["PROJECT_NAME"]
CLASSES = config["CLASSES"]
EFFICIENTNET_COMPOUND_COEF = config["EFFICIENTNET_COMPOUND_COEF"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCH_NUM = config["EPOCH_NUM"]
LEARNING_RATE = config["LEARNING_RATE"]
OPTIMIZER = config["OPTIMIZER"]
WEIGHT_PATH = config["WEIGHT_PATH"]
WEIGHT_PATH = WEIGHT_PATH.replace("d0", f"d{EFFICIENTNET_COMPOUND_COEF}")
ANCHOR_RATIOS = [eval(tup) for tup in config["ANCHOR_RATIOS"]]
ANCHOR_SCALES = [eval(expr) for expr in config["ANCHOR_SCALES"]]

DATASET_DIR = os.path.join(os.path.abspath(os.getcwd()), "datasets", PROJECT_NAME, "COCO")

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

model = EfficientDetBackbone(num_classes=len(CLASSES), compound_coef=EFFICIENTNET_COMPOUND_COEF, ratios=ANCHOR_RATIOS, scales=ANCHOR_SCALES)
try:
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(os.path.join("original_weights", f"efficientdet-d{EFFICIENTNET_COMPOUND_COEF}.pth")), strict=False)
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

loss_plot = []
for epoch in range(EPOCH_NUM):
    epoch_loss = []
    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):
        try:
            imgs, annot = data['img'], data['annot']
            imgs, annot = imgs.to(DEVICE), annot.to(DEVICE)

            optimizer.zero_grad()
            cls_loss, reg_loss = model(imgs, annot, obj_list=CLASSES)
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
    loss_plot.append(np.mean(epoch_loss))

epochs = range(1, len(loss_plot)+1)

torch.save(model.model.state_dict(), os.path.join("weights", WEIGHT_PATH))

fig = Figure(figsize=(18, 13))
canvas = FigureCanvas(fig)
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Loss vs Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.plot(epochs, loss_plot)
cv2.imwrite(os.path.join("LossGraphs", f"{WEIGHT_PATH[:-4]}.png"), fig_to_image(fig))