# CV Basics

## Photometric Loss ([photometric_loss.cu](photometric_loss.cu))
This project computes photometric (L2) loss between two images using CUDA for fast, parallel pixel-wise operations. Images are loaded and converted to grayscale with OpenCV, normalized to \([0, 1]\), and transferred to the GPU. A CUDA kernel calculates the squared difference at each pixel:  
$ L = \frac{1}{W \times H} \sum_{x,y} \| I_1(x, y) - I_2(x, y) \|^2 $,  
and the result is averaged to get the final loss. It's a lightweight and efficient setup useful for verifying vision models or stereo data, and matches PyTorch results with high precision.
