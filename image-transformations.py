# import numpy as np
import os
from PIL import Image
from os import listdir
import Augmentor


def resize_plus_change_type():
    from_path = './dataset'
    to_path = './dataset-transformed-28'
    for file in listdir(from_path):
        image = Image.open(from_path + '/' + file)
        if image.format == 'JPEG' or image.format == 'JPG':
            image_resized = image.resize((28, 28), resample=Image.BILINEAR)
            image_resized.save(to_path + '/' + os.path.splitext(file)[0] + '.png')
        else:
            pass
            # TODO: print name of image and check formats


def main():
    p = Augmentor.Pipeline('./dataset-transformed')
    p.rotate90(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.rotate180(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.rotate270(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.skew_tilt(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.shear(probability=1, max_shear_left=0.20, max_shear_right=0.20)
    p.status()
    p.process()

    p = Augmentor.Pipeline('./dataset-transformed')
    p.resize(probability=1, width=256, height=256)
    p.crop_random(probability=1, percentage_area=0.5)
    p.status()
    p.process()


if __name__ == '__main__':
    # resize_plus_change_type()
    main()
