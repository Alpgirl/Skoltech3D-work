#!/usr/bin/env python

from pathlib import Path
import subprocess

from reportlab.lib.pagesizes import mm
import numpy as np

from create_calibration_pattern import create_calibration_pattern


def make_calibration_pattern(
        tag_i,
        paper_size,
        display_size_mm,
        display_resolution,
        tag_img_root,
        patterns_dir,
        star_segments_n=16,
        square_size=1.2,
        tag_shift=(0, 0),
        tag_size=4,
        fullscreen=False
):
    args = (
        f' --tag36h11_path {tag_img_root}'
        f' --paper_size {paper_size}'
        f' --approx_square_length_in_cm {square_size}'
        f' --num_star_segments {star_segments_n}'
        f' --apriltag_index {tag_i}'
        f' --apriltag_shift_in_squares {tag_shift[0]} {tag_shift[1]}'
        f' --apriltag_length_in_squares {tag_size}'
        f' --output_dir {patterns_dir}'
    ).split()
    pagesize, out_file = create_calibration_pattern(args)

    display_size_mm = np.array(display_size_mm)
    display_resolution = np.array(display_resolution)
    pagesize_mm = np.array(pagesize) / mm

    pixels_per_mm = display_resolution / display_size_mm
    image_resolution = pixels_per_mm * pagesize_mm
    print(f'Pattern image resolution is {image_resolution[0]}x{image_resolution[1]}')
    image_resolution = image_resolution.round().astype(int)

    command = f'pdftoppm {out_file}.pdf -r 120 {out_file}'
    # print(command)
    subprocess.run(command.split())
    command = f'convert {out_file}-1.ppm -resize {image_resolution[0]}x{image_resolution[1]}! -quality 100 {out_file}.png'
    # print(command)
    subprocess.run(command.split())
    command = f'rm {out_file}-1.ppm'
    # print(command)
    subprocess.run(command.split())
    if fullscreen:
        command = f'convert {out_file}.png -gravity Center -crop {display_resolution[0]}x{display_resolution[1]}+0+0! -flatten  {out_file}_fs.png'
        # print(command)
        subprocess.run(command.split())
    return out_file


if __name__ == '__main__':
    tag_img_root = '/home/universal/.local/opt/apriltag-imgs/tag36h11'
    patterns_dir = '/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration/patterns'
    Path(patterns_dir).mkdir(exist_ok=True)

    # mode = 'Three displays'
    # mode = 'Single display large'
    # mode = 'Single display small'
    mode = 'Paper'

    if mode == 'Paper':
        # main pattern
        make_calibration_pattern(
            tag_i=0,
            paper_size='B3',
            square_size=3,
            tag_size=2,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # A3 pattern for IR on phones
        make_calibration_pattern(
            tag_i=10,
            paper_size='A3',
            square_size=5,
            tag_size=2,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # B3 pattern for IR on phones
        make_calibration_pattern(
            tag_i=20,
            paper_size='B3',
            square_size=5,
            tag_size=2,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
    elif mode == 'Single display small':
        make_calibration_pattern(
            tag_i=0,
            paper_size='B3',
            square_size=3,
            tag_size=2,
            fullscreen=True,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # make_calibration_pattern(
        #     tag_i=1,
        #     paper_size='B3',
        #     square_size=6,
        #     tag_size=2,
        #     fullscreen=True,
        #     display_size_mm=(296, 526.5),
        #     display_resolution=(1080, 1920),
        #     tag_img_root=tag_img_root,
        #     patterns_dir=patterns_dir,
        # )
        # make_calibration_pattern(
        #     tag_i=2,
        #     paper_size='B3',
        #     square_size=12,
        #     tag_size=2,
        #     fullscreen=True,
        #     display_size_mm=(296, 526.5),
        #     display_resolution=(1080, 1920),
        #     tag_img_root=tag_img_root,
        #     patterns_dir=patterns_dir,
        # )
    elif mode == 'Single display large':
        make_calibration_pattern(
            tag_i=0,
            paper_size='B3',
            tag_size=4,
            fullscreen=True,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
    elif mode == 'Three displays':
        # Middle display
        make_calibration_pattern(
            tag_i=0,
            paper_size='B3',
            tag_shift=(11, -11),
            tag_size=4,
            fullscreen=True,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # Left display
        make_calibration_pattern(
            tag_i=1,
            paper_size='B3',
            tag_shift=(15, 5),
            tag_size=4,
            fullscreen=True,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # Right display
        make_calibration_pattern(
            tag_i=2,
            paper_size='B3',
            tag_shift=(-2, 2),
            tag_size=4,
            fullscreen=True,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        # Aux pattern to counter a bug in calibration tool
        aux_pattern_file = make_calibration_pattern(
            tag_i=100,
            paper_size='B3',
            tag_size=0,
            display_size_mm=(296, 526.5),
            display_resolution=(1080, 1920),
            tag_img_root=tag_img_root,
            patterns_dir=patterns_dir,
        )
        aux_pattern_config = Path(f'{aux_pattern_file}.yaml')
        aux_pattern_config.rename(aux_pattern_config.parent / 'aux.yaml')
        for file in Path(aux_pattern_file).glob('*'):
            file.unlink(missing_ok=True)
