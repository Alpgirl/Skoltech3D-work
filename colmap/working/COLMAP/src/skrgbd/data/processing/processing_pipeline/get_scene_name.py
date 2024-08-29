from argparse import ArgumentParser

from skrgbd.data.dataset.dataset import wip_scene_name_by_id


def main():
    description = r"""Gets scene name by its id and prints it."""
    parser = ArgumentParser(description=description)
    parser.add_argument('--scene-i', type=int, required=True)
    args = parser.parse_args()

    scene_name = wip_scene_name_by_id[args.scene_i]
    print(scene_name)


if __name__ == '__main__':
    main()
