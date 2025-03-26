import numpy as np
from PIL import Image


def preprocess_image(img_array):
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    STD = np.array([0.26862954, 0.26130258, 0.27577711])

    img = Image.fromarray(img_array.astype("uint8"))
    img = resize_with_aspect_ratio(img, 224)
    img = center_crop(img, 224)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    return img_array.transpose(2, 0, 1)


def resize_with_aspect_ratio(image, target_size):
    original_width, original_height = image.size
    ratio = target_size / min(original_width, original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return image.resize((new_width, new_height), Image.BICUBIC)


def center_crop(image, target_size):
    width, height = image.size
    left = (width - target_size) / 2
    top = (height - target_size) / 2
    right = (width + target_size) / 2
    bottom = (height + target_size) / 2
    return image.crop((left, top, right, bottom))
