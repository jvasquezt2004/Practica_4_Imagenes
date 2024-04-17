import numpy as np
from PIL import Image, ImageStat


def calculate_model(images):
    image_quantity = len(images)
    sum_image = np.sum(np.stack([np.array(image) for image in images]), axis=0)
    mean_image = sum_image / image_quantity

    variance_image = (
        np.sum([np.square(np.array(image) - mean_image) for image in images], axis=0)
        / image_quantity
    )
    return Image.fromarray(np.uint8(mean_image)), Image.fromarray(
        np.uint8(variance_image)
    )


image_paths = ["img/Background1.jpeg", "img/Background2.jpeg", "img/Background3.jpeg"]
background_images = [Image.open(path) for path in image_paths]

mean_image, variance_image = calculate_model(background_images)

mean_image.show()
variance_image.show()
