import os
from PIL import Image
from os import listdir
import Augmentor

from_path = './datasets/flowers'
to_path = './datasets/flowers64'


def crop_scale():
    for file in listdir(from_path):
        image = Image.open(from_path + '/' + file)

        if image.format != 'JPEG' and image.format != 'JPG' and image.format != 'PNG':
            print('Error, incorrect format, file: ', file)

        width, height = image.size

        # get smallest size, to make square
        square_len = min(width, height)

        # calculate crop coordinates (square should be in the center)
        left = (width - square_len) / 2
        top = (height - square_len) / 2
        right = (width + square_len) / 2
        bottom = (height + square_len) / 2

        # crop square
        image = image.crop((left, top, right, bottom))

        # scale and save
        image_resized = image.resize((64, 64), resample=Image.BILINEAR)
        image_resized.save(to_path + '/' + os.path.splitext(file)[0] + '.png')


def apply_transformations():
    # p = Augmentor.Pipeline(to_path)
    # p.rotate90(probability=1)
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

    # not sure, whether crop is needed
    # p = Augmentor.Pipeline(to_path)
    # p.resize(probability=1, width=256, height=256)
    # p.crop_random(probability=1, percentage_area=0.5)
    # p.status()
    # p.process()


if __name__ == '__main__':
    crop_scale()
    apply_transformations()
