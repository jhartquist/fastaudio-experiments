from fastai.vision.all import *
from fastaudio.core.all import *

import wandb
from fastai.callback.wandb import *


path = untar_data(URLs.ESC50)
model = resnet18(pretrained=True)
