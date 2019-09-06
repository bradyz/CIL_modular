from pathlib import Path
from collections import namedtuple

import pickle

import numpy as np
import torch
import cv2

from utils import mapping_helper
from learn_by_cheat import video_maker


N_STEPS = 5
N_SKIP = 20
DEBUG = False


def _get_lat_lon_std(sol):
    sp = sol.split('WGS84')[1].strip().split()

    return float(sp[0]), float(sp[1])


def get_images(video_path, capture_every=1):
    capture = cv2.VideoCapture(video_path)
    index_frame = list()
    index = -1

    while capture.isOpened():
        ret, frame = capture.read()
        index += 1

        if not ret:
            break
        elif index % capture_every != 0:
            continue

        w = frame.shape[1] // 3
        index_frame.append((index, frame[:, w:w+w]))

    return index_frame


class RFSDataset(torch.utils.data.Dataset):
    Info = namedtuple(
            'Info',
            ['x', 'y', 'orientation', 'speed',
                'dgps', 'lat_std', 'lon_std'])

    def __init__(self, directory):
        video_path = Path(directory) / 'video.avi'
        info_path = Path(directory) / 'video.pkl'

        assert video_path.exists()
        assert info_path.exists()

        self._index_frame = get_images(str(video_path))
        self._info = pickle.load(open(str(info_path), 'rb'))
        self._map = mapping_helper.mapping_helper(output_height_pix=256, version='v1')

    def __len__(self):
        return len(self._index_frame)

    def _get_info(self, index):
        speed, _, pos, ori, dgps, nmea = self._info[index]
        lat_std, lon_std = _get_lat_lon_std(nmea[-1])

        return self.Info(
                pos[0], pos[1], ori, speed,
                dgps, lat_std, lon_std)

    def __getitem__(self, idx):
        """
        speed m/s
        orientation radians
        lat/lon std in cm / 100
        """
        index, frame = self._index_frame[idx]
        info = self._get_info(index)
        map_view = self._map.get_map('rfs', info.dgps, info.orientation)

        x_pixel, y_pixel = self._map.loc_to_pix_rfs((info.x, info.y))
        ox, oy = np.cos(info.orientation), np.sin(info.orientation)
        R = np.array([
            [ox, -oy],
            [oy,  ox]])

        locations = np.zeros((N_STEPS, 2), dtype=np.float32)

        for i in range(N_STEPS):
            info_i = self._get_info(index + N_SKIP * i)
            x_pixel_i, y_pixel_i = self._map.loc_to_pix_rfs((info_i.x, info_i.y))

            u = R.dot([x_pixel_i - x_pixel, y_pixel_i - y_pixel])
            locations[i] = u

            i = int(map_view.shape[0] - 1 + u[0])
            j = int(map_view.shape[1] // 2 + u[1])

            if DEBUG:
                map_view[i-2:i+2,j-2:j+2] = 0.5

        if DEBUG:
            video_maker.add(255 * np.stack(3 * [map_view], -1))

        return map_view, locations


if __name__ == '__main__':
    DEBUG = True
    video_maker.init()

    #data = RFSDataset('/home/bradyzhou/data/mkz_2019.8.23/2019-08-23_23-39-53')
    data = RFSDataset('/home/bradyzhou/data/mkz_2019.8.23/2019-08-23_23-12-10')

    for i in range(len(data)):
        data[i]
