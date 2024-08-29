from copy import deepcopy
from pathlib import Path
import struct

import numpy as np


class Dataset:
    r"""

    Attributes
    ----------
    cameras_n : int
        Number of cameras.
    image_sizes : list of tuple of int
        of shape [cameras_n, 2]. (h, w) image resolution for each camera.
    pattern_data : list of dicts
        cell_size: float
        features_id: np.ndarray of shape [features_n]
        features_xy: np.ndarray of shape [features_n, 2]
    image_data : dict
        image_name: list of dicts shape [cameras_n]
            features_id: np.ndarray of shape [features_n_in_im_cam]
            features_xy: np.ndarray of shape [features_n_in_im_cam, 2]
    """
    cameras_n = None
    image_sizes = None
    image_data = None
    pattern_data = None

    @classmethod
    def fromfile(cls, file):
        self = cls()
        with open(file, 'rb') as file:
            # Converters
            u32_reader = struct.Struct('!I')
            f32_reader = struct.Struct('f')  # yes, they don't swap bytes for floats

            def read_u32():
                return u32_reader.unpack(file.read(4))[0]

            def read_f32():
                return f32_reader.unpack(file.read(4))[0]

            # File format identifier
            header = file.read(10).decode()
            if header != 'calib_data':
                raise RuntimeError('Wrong header')
            version = read_u32()
            if version != 0:
                raise RuntimeError('Wrong version')

            # Cameras
            self.cameras_n = read_u32()
            self.image_sizes = []
            for camera_i in range(self.cameras_n):
                w = read_u32()
                h = read_u32()
                self.image_sizes.append((h, w))

            # Imagesets
            images_n = read_u32()
            self.image_data = dict()
            for image_i in range(images_n):
                filename_len = read_u32()
                filename = file.read(filename_len).decode()
                self.image_data[filename] = image_data = []
                for camera_i in range(self.cameras_n):
                    features_n = read_u32()
                    features_xy_id = np.fromfile(file, dtype=np.float32, count=features_n * 3).reshape(features_n, 3)
                    features_id = features_xy_id[:, 2]
                    features_xy = features_xy_id[:, :2]
                    features_id = np.array(struct.unpack(f'!{features_n}I', features_id.tobytes()), dtype=np.uint32)
                    image_data.append(dict(features_id=features_id, features_xy=features_xy))

            # Known geometries
            patterns_n = read_u32()
            self.pattern_data = []
            for pattern_i in range(patterns_n):
                cell_size = read_f32()
                features_n = read_u32()
                n = features_n * 3
                features_id_xy = struct.unpack(f'!{n}i', file.read(n * 4))
                features_id_xy = np.array(features_id_xy, dtype=np.int32).reshape(-1, 3)
                features_id = features_id_xy[:, 0]
                features_xy = features_id_xy[:, 1:]
                self.pattern_data.append(dict(cell_size=cell_size, features_id=features_id, features_xy=features_xy))
        return self

    @classmethod
    def merge_cameras(cls, dataset_bins, ignore_file_extensions=True):
        r"""Merges datasets calculated for different cameras for the same calibration patterns.

        Parameters
        ----------
        dataset_bins : iterable of str
        ignore_file_extensions : bool
        """
        datasets = list(map(cls.fromfile, dataset_bins))
        merged_dataset = cls()

        merged_dataset.pattern_data = datasets[0].pattern_data
        for dataset in datasets:
            if not pattern_data_eq(dataset.pattern_data, merged_dataset.pattern_data):
                raise ValueError('Different pattern data')

        merged_dataset.cameras_n = sum(dataset.cameras_n for dataset in datasets)
        merged_dataset.image_sizes = [size for dataset in datasets for size in dataset.image_sizes]

        if ignore_file_extensions:
            filenames = set(str(Path(filename).stem) for dataset in datasets for filename in dataset.image_data)
            image_datas = []
            for dataset in datasets:
                image_datas.append({str(Path(filename).stem): v for (filename, v) in dataset.image_data.items()})
        else:
            filenames = set(filename for dataset in datasets for filename in dataset.image_data)
            image_datas = [dataset.image_data for dataset in datasets]

        merged_dataset.image_data = dict()
        for filename in filenames:
            merged_image_data = []
            for dataset, image_data in zip(datasets, image_datas):
                if filename in image_data:
                    merged_image_data.extend(image_data[filename])
                else:
                    merged_image_data.extend([dict(features_id=np.empty([0], dtype=np.uint32),
                                                   features_xy=np.empty([0, 2], dtype=np.float32))] * dataset.cameras_n)
            merged_dataset.image_data[filename] = merged_image_data
        return merged_dataset

    @classmethod
    def merge_positions(cls, dataset_bins):
        r"""Merges datasets calculated for different positions for the same cameras and calibration patterns.

        Parameters
        ----------
        dataset_bins : iterable of str
        """
        datasets = list(map(cls.fromfile, dataset_bins))
        merged_dataset = cls()

        merged_dataset.pattern_data = datasets[0].pattern_data
        if any(not pattern_data_eq(dataset.pattern_data, merged_dataset.pattern_data) for dataset in datasets):
            raise ValueError('Different pattern data')

        merged_dataset.cameras_n = datasets[0].cameras_n
        merged_dataset.image_sizes = datasets[0].image_sizes
        if any(not camera_data_eq(dataset, merged_dataset) for dataset in datasets):
            raise ValueError('Different camera data')

        merged_dataset.image_data = dict()
        for dataset in datasets:
            repeated_poses = set(merged_dataset.image_data.keys()).intersection(set(dataset.image_data.keys()))
            if len(repeated_poses) > 0:
                raise ValueError(f'Repeated poses {repeated_poses}')
            merged_dataset.image_data.update(dataset.image_data)
        return merged_dataset

    def save(self, file):
        with open(file, 'wb') as file:
            # Converters
            u32_writer = struct.Struct('!I')

            def write_u32(v):
                return file.write(u32_writer.pack(v))

            file.write('calib_data'.encode())
            write_u32(0)  # version

            # Cameras
            write_u32(self.cameras_n)
            for (h, w) in self.image_sizes:
                write_u32(w)
                write_u32(h)

            # Imagesets
            write_u32(len(self.image_data))
            for filename, image_data in self.image_data.items():
                write_u32(len(filename))
                file.write(filename.encode())
                for per_camera_data in image_data:
                    features_n = len(per_camera_data['features_id'])
                    write_u32(features_n)
                    ids = struct.pack(f'!{features_n}I', *per_camera_data['features_id'])
                    ids = np.frombuffer(ids, dtype=np.float32)
                    xyi = np.concatenate([per_camera_data['features_xy'], ids[:, None]], 1)
                    xyi.tofile(file)

            # Known geometries
            write_u32(len(self.pattern_data))
            for pattern_data in self.pattern_data:
                file.write(struct.pack('f', pattern_data['cell_size']))
                features_n = len(pattern_data['features_id'])
                write_u32(features_n)
                ixy = np.concatenate([pattern_data['features_id'][:, None], pattern_data['features_xy']], 1)
                ixy = struct.pack(f'!{features_n * 3}i', *ixy.ravel())
                file.write(ixy)

    def merge_boards(self, board_poses):
        r"""For each camera position recombines feature detections for different positions of a single calibration board
        as if the calibration board in a certain position was a different calibration board.

        Parameters
        ----------
        board_poses : sequence of str
        """
        if len(self.pattern_data) != 1:
            raise NotImplementedError(f'Not implemented for datasets with more than one pattern')
        board_poses_n = len(board_poses)
        features_n = len(self.pattern_data[0]['features_id'])
        cams_n = len(next(iter(self.image_data.values())))

        # Replicate the calibration board for each board position
        merged_pattern_data = []
        for board_pos_i in range(board_poses_n):
            pattern_replica = deepcopy(self.pattern_data[0])
            pattern_replica['features_id'] += features_n * board_pos_i
            merged_pattern_data.append(pattern_replica)

        # Collect detections for each camera positions and different board positions
        merged_image_data = dict()
        while len(self.image_data.keys()) > 0:
            pos_filename = next(iter(self.image_data.keys()))
            for board_pos in board_poses:
                if pos_filename.startswith(board_pos + '_'):
                    break
            filename = pos_filename[len(board_pos) + 1:]  # pos_filename == {some_board_pos}_{camera_pos}

            merged_data = []
            for cam_i in range(cams_n):
                features_id = []
                features_xy = []
                for board_pos_i, board_pos in enumerate(board_poses):
                    pos_filename = f'{board_pos}_{filename}'
                    if pos_filename in self.image_data:
                        cam_data = self.image_data[pos_filename][cam_i]
                        features_id.append(cam_data['features_id'] + features_n * board_pos_i)
                        features_xy.append(cam_data['features_xy'])
                merged_data.append({'features_id': np.concatenate(features_id),
                                    'features_xy': np.concatenate(features_xy, 0)})
            merged_image_data[filename] = merged_data

            for board_pos_i, board_pos in enumerate(board_poses):
                pos_filename = f'{board_pos}_{filename}'
                if pos_filename in self.image_data:
                    del self.image_data[pos_filename]

        self.pattern_data = merged_pattern_data
        self.image_data = merged_image_data


def pattern_data_eq(d1, d2):
    if len(d1) != len(d2):
        return False
    if any(((p1['cell_size'] != p2['cell_size'])
            or (p1['features_id'] != p2['features_id']).any()
            or (p1['features_xy'] != p2['features_xy']).any())
           for (p1, p2) in zip(d1, d2)):
        return False
    return True


def camera_data_eq(d1, d2):
    if d1.cameras_n != d2.cameras_n:
        return False
    if any(d1.image_sizes[i] != d2.image_sizes[i] for i in range(d1.cameras_n)):
        return False
    return True
