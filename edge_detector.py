# Converting the image to grayscale.
import cv2
import numpy as np
from skimage import io, color, feature
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# import scipy
# from scipy import ndimage
# from PIL import Image



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
  # edges_filtered = cv2.Canny(edges_foreground, 20, 50) #30, 100) # adjust the parameters of the canny here

  edges_filtered = cv2.Canny(edges_foreground, 5, 10) #for gtsrb

  # Crop off the edges out of the moving area
  cropped = (foreground // 255) * edges_filtered

  return cropped


def compute_energy_matrix(img): 
    '''
    extract the sobel edge detector
    '''
    gray = np.array(img.mean(axis=2).cpu()*255).astype('uint8')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Compute X derivative of the image 
    sobel_x = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=3) 
 
    # Compute Y derivative of the image 
    sobel_y = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=3) 
 
    abs_sobel_x = cv2.convertScaleAbs(sobel_x) 
    abs_sobel_y = cv2.convertScaleAbs(sobel_y) 
 
    # Return weighted summation of the two images i.e. 0.5*X + 0.5*Y 
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0) 
 
# Find vertical seam in the input image 

def detect_edge_sobel(img):
  # image = np.array(img.mean(axis=2).cpu()*255).astype('uint8')
  # processed = ndimage.sobel(img, 0)

  gray = np.array(img.mean(axis=2).cpu()*255).astype('uint8')
  imgBLR = cv2.GaussianBlur(gray, (3,3), 0)
  # imgEDG = cv2.Canny(imgBLR, 30, 150) 


  # sobel
  img_sobelx = cv2.Sobel(imgBLR,cv2.CV_8U,1,0,ksize=3)
  img_sobely = cv2.Sobel(imgBLR,cv2.CV_8U,0,1,ksize=3)
  imgEDG = img_sobelx + img_sobely

  return imgEDG

  # image = image.convert("L")
  # image = image.filter(ImageFilter.FIND_EDGES) 
  # return processed
  # scipy.misc.imsave('sobel.jpg', processed)


def detect_edge_mnist(img):
    edge_map = feature.canny(np.array(img[0], dtype=np.float64), sigma = .1, low_threshold=1.5) #, high_threshold=.1)
    return edge_map[None]    