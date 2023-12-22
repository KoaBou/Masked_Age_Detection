from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
from mivolo.model.mivolo_model import *
import torch

# Choose a MiVOLO variant
model_variant = "mivolo_d1_224"

# Create the MiVOLO model
model = mivolo_d1_224(pretrained=False)  # Set pretrained=True if you want to use pre-trained weights

# Set the model to evaluation mode
model.eval()

# Input tensor (batch size of 1, 3 channels, image size 224x224)
input_tensor = torch.randn(1, 3, 224, 224)
masked = False

# Forward pass
with torch.no_grad():
    output = model(input_tensor, masked)

# Print the output shape
print("Output shape:", output)