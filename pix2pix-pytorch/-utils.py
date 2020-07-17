import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def detect_edge_batch(imgs):        
    # YOU MAY NEED TO MODIFY THIS FUNCTION IN ORDER TO CHOOSE THE BEST EDGE DETECTION THAT WORKS ON YOUR DATA
    # FOR THAT, YOU MAY ALSO NEED TO CHANGE THE SOME PARAMETERS; SEE EDGE_DETECTOR.PY
    # import pdb; pdb.set_trace()

    for im in imgs:
        edge_map = edge_detect(im) 
        # edge_map = edge_map/255.
        if (edge_map.max() - edge_map.min()) > 0:
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        
        edge_map = torch.tensor(edge_map, dtype=torch.float32)
        im[-1] = edge_map # replace the last map
    
    return imgs
