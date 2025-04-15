import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

to_tensor = T.Compose([
    T.Grayscale(),
    T.ToTensor(),          # Converts to [0,1] float32 and shape [1, H, W]
])

img_left = to_tensor(Image.open("images/L.png").convert("RGB"))
img_right = to_tensor(Image.open("images/R.png").convert("RGB"))

# Flatten images
left_flat = img_left.view(-1)
right_flat = img_right.view(-1)

# Compute per-pixel squared difference
diff = left_flat - right_flat
loss = torch.sum(diff ** 2) / (img_left.shape[2] * img_left.shape[1])

print(f"Photometric L2 Loss: {loss.item():.6f}")
