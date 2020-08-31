# Converting the image to grayscale.
import cv2
import numpy as np
from skimage import io, color, feature
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_edge_new(img):
  fgbg = cv2.createBackgroundSubtractorMOG2(
        history=10,
        varThreshold=2,
        detectShadows=False)

  gray = np.array(img.mean(axis=2).cpu()*255).astype('uint8')

  # Extract the foreground
  edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
  foreground = fgbg.apply(edges_foreground)

  # Smooth out to get the moving area
  kernel = np.ones((50,50),np.uint8)
  foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

  # Applying static edge extraction
  edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
  edges_filtered = cv2.Canny(edges_foreground, 30, 100)

  # Crop off the edges out of the moving area
  cropped = (foreground // 255) * edges_filtered

  return cropped#edges_filtered



def detect_edge_mnist(img):
    edge_map = feature.canny(np.array(img[0], dtype=np.float64), sigma = .1, low_threshold=1.5) #, high_threshold=.1)
    return edge_map[None]    





def detect_edge_new_cifar(img):
  img = img.permute(1,2,0)
  gray = np.array(img.mean(axis=2)*255).astype('uint8')
  imgBLR = cv2.GaussianBlur(gray, (5,5), 3)
  imgEDG = cv2.Canny(imgBLR, 40, 150)
  if (imgEDG.max() - imgEDG.min()) > 0:
    imgEDG = (imgEDG - imgEDG.min()) / (imgEDG.max() - imgEDG.min())

  imgEDG = imgEDG/255.

  return imgEDG

