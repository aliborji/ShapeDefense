from lib import *
import edge_detector


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# This specifies whether edge detection should be done only on RGB (when FALSE) or use 
# False means only rgb is used for edge detection
EDGE_ALL_CHANNELS = False

# edge_detect = None

edge_detect = edge_detector.detect_edge_gtsrb

# edge_detect = edge_detector.detect_edge_mnist

# edge_detect = edge_detector.detect_edge_new

# edge_detect = edge_detector.detect_edge_tiny

# edge_detect = edge_detector.compute_energy_matrix
# edge_detect = edge_detector.detect_edge_fashionmnist
# edge_detect = edge_detector.detect_edge_mnist

# edge_detect = edge_detector.detect_edge_sobel


# edge_detect = edge_detector.detect_edge_new_cifar






# edge detectors used: (for replication)
# detect_edge_new for DogVsCat  canny   !
# detect_edge_mnist for mnist and fashionMNIST
# compute_energy_matrix for dogs breeds
# detect_edge_gtsrb for gtsrb and icons, sketch
# cifar detect_edge_new_cifar OR detect_edge_tiny
# imanette detect_edge_tiny
# tiny imagenet detect_edge_tiny