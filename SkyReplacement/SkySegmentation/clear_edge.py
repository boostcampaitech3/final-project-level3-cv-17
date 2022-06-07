import numpy as np
import cv2


def edge_mask(mask):
    mask = np.asarray(mask, dtype=np.double)
    gx, gy = np.gradient(mask)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge >= 0.5] = 1.0
    temp_edge[temp_edge < 0.5] = 0.0
    temp_edge = np.asarray(temp_edge, dtype=np.uint8)
    return temp_edge

def variance_filter(img_gray, window_size = 3): 
    """
    Variance filter
    Calculate the variance in the sliding window
    """
    img_gray = img_gray.astype(np.float64)
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (window_size, window_size), borderType=cv2.BORDER_REPLICATE) for x in (img_gray, img_gray*img_gray))
    return wsqrmean - wmean*wmean
