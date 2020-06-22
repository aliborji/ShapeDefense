from lib import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

WIDTH = 150
HEIGHT = 150

BATCH_SIZE = 100
NUM_EPOCHS = 100
MEAN = (1./255,1./255,1./255)
STD = (1.0,1.0,1.0)

# save_path_rgb = './dogs_cats_rgb.pth'
# save_path_edge = './dogs_cats_edge.pth'
# save_path_rgbedge = './dogs_cats_rgbedge.pth'

# save_path_rgb_robust = './dogs_cats_rgb_robust.pth'
# save_path_edge_robust = './dogs_cats_edge_robust.pth'
# save_path_rgbedge_robust = './dogs_cats_rgbedge_robust.pth'
