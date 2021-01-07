import os
from PIL import Image
from os import listdir
import Augmentor

from_path = './datasets/fauvism'
to_path = './datasets/fauvism64'


def resize_plus_change_type():
    for file in listdir(from_path):
        image = Image.open(from_path + '/' + file)
        if image.format == 'JPEG' or image.format == 'JPG':
            image_resized = image.resize((64, 64), resample=Image.BILINEAR)
            image_resized.save(to_path + '/' + os.path.splitext(file)[0] + '.png')
        else:
            pass
            # TODO: print name of image and check formats


def apply_transformations():
    # p = Augmentor.Pipeline(to_path)
    # p.rotate90(probability=1)
    # p.status()
    # p.process()
    #
    # p = Augmentor.Pipeline(to_path)
    # p.rotate180(probability=1)
    # p.status()
    # p.process()
    #
    # p = Augmentor.Pipeline(to_path)
    # p.rotate270(probability=1)
    # p.status()
    # p.process()

    p = Augmentor.Pipeline(to_path)
    p.flip_left_right(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline(to_path)
    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    p.status()
    p.process()

    p = Augmentor.Pipeline(to_path)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.status()
    p.process()

    p = Augmentor.Pipeline(to_path)
    p.skew_tilt(probability=1)
    p.status()
    p.process()

    p = Augmentor.Pipeline(to_path)
    p.shear(probability=1, max_shear_left=0.20, max_shear_right=0.20)
    p.status()
    p.process()

    # TODO: ?
    p = Augmentor.Pipeline(to_path)
    p.resize(probability=1, width=256, height=256)
    p.crop_random(probability=1, percentage_area=0.5)
    p.status()
    p.process()


if __name__ == '__main__':
    resize_plus_change_type()
    apply_transformations()
