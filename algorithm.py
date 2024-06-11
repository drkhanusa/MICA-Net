import numpy as np
import pandas as pd
import cv2
from typing import List, Generator, Union
from abc import ABC
from collections.abc import Sequence
import types
from math import sqrt

__all__ = [
    'FullSampler',
    'SystematicSampler',
    'RandomSampler',
    'OnceRandomSampler',
    'RandomTemporalSegmentSampler',
    'OnceRandomTemporalSegmentSampler',
    'LambdaSampler',
    'synchronize_state',
]


class _MediaCapture:
    def __init__(self, source):
        self.source = source
        if isinstance(source, Sequence) and not isinstance(source, str):
            self.paths = list(source)
            self.is_video = False
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_video = True
        self._frame_id = 0

    @classmethod
    def from_video_capture(cls, cap):
        raise NotImplementedError

    def is_opened(self):
        if self.is_video:
            return self.cap.isOpened()
        else:
            return len(self.paths) > 0

    def get(self, prop):
        if self.is_video:
            return self.cap.get(prop)

    def set(self, prop, value):
        if self.is_video:
            return self.cap.set(prop, value)

    def read(self):
        if self.is_video:
            self._frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, frame = self.cap.read()
        else:
            frame = cv2.imread(self.paths[self._frame_id])
            ok = frame is not None
        self._frame_id += 1
        return ok, frame

    def release(self):
        if self.is_video:
            self.cap.release()
        else:
            self.paths.clear()

    def seek(self, frame_id):
        if self.is_video:
            if frame_id == self._frame_id:
                return
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self._frame_id = frame_id

    @property
    def frame_count(self):
        if self.is_video:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return len(self.paths)

    @property
    def fps(self):
        if self.is_video:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.

    @property
    def frame_id(self):
        return self._frame_id

    def sample(self, frame_ids):
        frames = []
        for frame_id in frame_ids:
            self.seek(frame_id)
            ok, frame = self.read()
            if not ok:
                if self.is_video:
                    raise RuntimeError(f'Unable to read frame {frame_id} of {self.source}.')
                else:
                    raise RuntimeError(f'Unable to read file {self.paths[frame_id]}.')
            frames.append(frame)
        return frames

    def __str__(self):
        ret = f'{self.__class__.__name__}'
        ret += f'(source="{self.source}")' if self.is_video else f'(source={self.source})'
        return ret


class _BaseSampler(ABC):
    def __init__(self, n_frames=16):
        if not n_frames:
            raise ValueError(f'n_frames must be positive number, got {n_frames}.')
        self.n_frames = n_frames
        self._presampling_hooks = []

    def __call__(self, source, start_frame=None, end_frame=None, sample_id=None):
        cap = _MediaCapture(source)
        if not cap.is_opened():
            raise RuntimeError(f'{source} is invalid.')
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = cap.frame_count - 1
        elif end_frame > cap.frame_count - 1:
            end_frame = cap.frame_count - 1

        for hook in self._presampling_hooks:
            hook(source, start_frame, end_frame, sample_id)
        sampled_frame_ids = self._get_sampled_frame_ids(source, start_frame, end_frame, sample_id)
        return cap.sample(sampled_frame_ids)

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id):
        raise NotImplementedError

    def register_presampling_hook(self, hook):
        self._presampling_hooks.append(hook)

    def clear_presampling_hooks(self):
        self._presampling_hooks.clear()


class _BaseMemorizedSampler(_BaseSampler, ABC):
    def __init__(self, n_frames=16):
        super(_BaseMemorizedSampler, self).__init__(n_frames)
        self.memory = {}

    def __call__(self, source, start_frame=None, end_frame=None, sample_id=None):
        if sample_id is None:
            raise RuntimeError('sample_id is required.')
        return super(_BaseMemorizedSampler, self).__call__(source, start_frame, end_frame, sample_id)

    def clear(self):
        self.memory.clear()


class FullSampler(_BaseSampler):
    """Sample all frames"""

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        return list(range(start_frame, end_frame))


class SystematicSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        sampled_frame_ids = np.linspace(start_frame, end_frame, self.n_frames)
        return sampled_frame_ids.round().astype(np.int64)


class RandomSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        sampled_frame_ids = start_frame + np.random.rand(self.n_frames) * (end_frame - start_frame)
        sampled_frame_ids.sort()
        return sampled_frame_ids.round().astype(np.int64)


class OnceRandomSampler(_BaseMemorizedSampler, RandomSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        if sample_id in self.memory:
            return self.memory[sample_id]
        sampled_frame_ids = RandomSampler._get_sampled_frame_ids(self, source, start_frame, end_frame)
        self.memory[sample_id] = sampled_frame_ids
        return sampled_frame_ids


class RandomTemporalSegmentSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        segments = np.linspace(start_frame, end_frame, self.n_frames + 1)
        segment_length = (end_frame - start_frame) / self.n_frames
        sampled_frame_ids = segments[:-1] + np.random.rand(self.n_frames) * segment_length
        return sampled_frame_ids.round().astype(np.int64)


class OnceRandomTemporalSegmentSampler(_BaseMemorizedSampler, RandomTemporalSegmentSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        if sample_id in self.memory:
            return self.memory[sample_id]
        sampled_frame_ids = RandomTemporalSegmentSampler._get_sampled_frame_ids(self, source, start_frame, end_frame)
        self.memory[sample_id] = sampled_frame_ids
        return sampled_frame_ids


class LambdaSampler(_BaseSampler):
    def __init__(self, get_sampled_frame_ids_func):
        super(LambdaSampler, self).__init__(n_frames=0)
        self.get_sampled_frame_ids_func = get_sampled_frame_ids_func

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        return self.get_sampled_frame_ids_func(source, start_frame, end_frame)


class synchronize_state:
    def __init__(self, samplers: Union[List[_BaseSampler], Generator]):
        if isinstance(samplers, types.GeneratorType):
            samplers = list(samplers)
        self.samplers = samplers
        self._random_state = None

    def __enter__(self):
        self._random_state = np.random.get_state()
        for sampler in self.samplers:
            sampler.register_presampling_hook(self._reuse_numpy_state)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for sampler in self.samplers:
            sampler.clear_presampling_hooks()
        self._random_state = None

    def _reuse_numpy_state(self, *args, **kwargs):
        np.random.set_state(self._random_state)


def GramianAngularSumamationField(data_path):
    data = pd.read_csv(data_path, header=None).values
    ax_sub, ay_sub, az_sub, gx_sub, gy_sub, gz_sub = np.array(data[:, 0], float), np.array(data[:, 1], float), np.array(data[:, 2], float), np.array(data[:, 3], float), np.array(data[:, 4], float), np.array(data[:, 5], float)


    mx_sub, my_sub = [], []
    for abc in range(len(ax_sub)):
        mx_sub.append(sqrt(ax_sub[abc] * ax_sub[abc] + ay_sub[abc] * ay_sub[abc] + az_sub[abc] * az_sub[abc]))
        my_sub.append(sqrt(gx_sub[abc] * gx_sub[abc] + gy_sub[abc] * gy_sub[abc] + gz_sub[abc] * gz_sub[abc]))

    max_ax_sub, max_ay_sub, max_az_sub, min_ax_sub, min_ay_sub, min_az_sub = np.max(ax_sub), np.max(ay_sub), np.max(
        az_sub), np.min(
        ax_sub), np.min(ay_sub), np.min(az_sub)
    max_gx_sub, max_gy_sub, max_gz_sub, min_gx_sub, min_gy_sub, min_gz_sub = np.max(gx_sub), np.max(gy_sub), np.max(
        gz_sub), np.min(
        gx_sub), np.min(gy_sub), np.min(gz_sub)
    max_mx_sub, max_my_sub, min_mx_sub, min_my_sub = np.max(mx_sub), np.max(my_sub), np.min(
        mx_sub), np.min(my_sub)

    t = len(ax_sub)
    for a in range(t):
        ax_sub[a] = (ax_sub[a] - min_ax_sub) / (max_ax_sub - min_ax_sub)
        ay_sub[a] = (ay_sub[a] - min_ay_sub) / (max_ay_sub - min_ay_sub)
        az_sub[a] = (az_sub[a] - min_az_sub) / (max_az_sub - min_az_sub)

        gx_sub[a] = (gx_sub[a] - min_gx_sub) / (max_gx_sub - min_gx_sub)
        gy_sub[a] = (gy_sub[a] - min_gy_sub) / (max_gy_sub - min_gy_sub)
        gz_sub[a] = (gz_sub[a] - min_gz_sub) / (max_gz_sub - min_gz_sub)

        mx_sub[a] = (mx_sub[a] - min_mx_sub) / (max_mx_sub - min_mx_sub)
        my_sub[a] = (my_sub[a] - min_my_sub) / (max_my_sub - min_my_sub)
        # mz_sub[a] = (mz_sub[a] - min_mz_sub) / (max_mz_sub - min_mz_sub)

    phi_ax, phi_ay, phi_az, phi_gx, phi_gy, phi_gz, phi_mx, phi_my, phi_mz = np.zeros(t, dtype=float), np.zeros(t,
                                                                                                                dtype=float), np.zeros(
        t, dtype=float), np.zeros(t, dtype=float), np.zeros(t, dtype=float), np.zeros(t, dtype=float), np.zeros(t,
                                                                                                                dtype=float), np.zeros(
        t, dtype=float), np.zeros(t, dtype=float)

    for a in range(len(ax_sub)):
        phi_ax[a] = np.arcsin(ax_sub[a])
        phi_ay[a] = np.arcsin(ay_sub[a])
        phi_az[a] = np.arcsin(az_sub[a])

        phi_gx[a] = np.arcsin(gx_sub[a])
        phi_gy[a] = np.arcsin(gy_sub[a])
        phi_gz[a] = np.arcsin(gz_sub[a])

        phi_mx[a] = np.arcsin(mx_sub[a])
        phi_my[a] = np.arcsin(my_sub[a])
        # phi_mz[a] = np.arcsin(mz_sub[a])

    ax_matrix, ay_matrix, az_matrix, gx_matrix, gy_matrix, gz_matrix, mx_matrix, my_matrix, mz_matrix = np.zeros(
        (t, t)), np.zeros(
        (t, t)), np.zeros(
        (t, t)), np.zeros((t, t)), np.zeros((t, t)), np.zeros((t, t)), np.zeros((t, t)), np.zeros((t, t)), np.zeros(
        (t, t))
    for i in range(t):
        for j in range(t):
            ax_matrix[i, j] = phi_ax[i] - phi_ax[j]
            ay_matrix[i, j] = phi_ay[i] - phi_ay[j]
            az_matrix[i, j] = phi_az[i] - phi_az[j]

            gx_matrix[i, j] = phi_gx[i] - phi_gx[j]
            gy_matrix[i, j] = phi_gy[i] - phi_gy[j]
            gz_matrix[i, j] = phi_gz[i] - phi_gz[j]

            mx_matrix[i, j] = phi_mx[i] - phi_mx[j]
            my_matrix[i, j] = phi_my[i] - phi_my[j]
            # mz_matrix[i, j] = phi_mz[i] - phi_mz[j]

    ax_matrix = np.sin(ax_matrix)
    ay_matrix = np.sin(ay_matrix)
    az_matrix = np.sin(az_matrix)

    gx_matrix = np.sin(gx_matrix)
    gy_matrix = np.sin(gy_matrix)
    gz_matrix = np.sin(gz_matrix)

    mx_matrix = np.sin(mx_matrix)
    my_matrix = np.sin(my_matrix)
    # mz_matrix = np.sin(mz_matrix)

    ax_matrix = cv2.resize(ax_matrix, (224, 224))
    ay_matrix = cv2.resize(ay_matrix, (224, 224))
    az_matrix = cv2.resize(az_matrix, (224, 224))
    gx_matrix = cv2.resize(gx_matrix, (224, 224))
    gy_matrix = cv2.resize(gy_matrix, (224, 224))
    gz_matrix = cv2.resize(gz_matrix, (224, 224))
    mx_matrix = cv2.resize(mx_matrix, (224, 224))
    my_matrix = cv2.resize(my_matrix, (224, 224))

    result = np.dstack((ax_matrix, ay_matrix, az_matrix, gx_matrix, gy_matrix, gz_matrix, mx_matrix, my_matrix))
    result = result.transpose(2, 0, 1)
    result = np.expand_dims(result, axis=0)
    return np.ascontiguousarray(result.astype(np.float32))


def pre_processing(video_file, sample_id, transform):
    X = RandomTemporalSegmentSampler(n_frames=16)(video_file, sample_id=None)
    X = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in X]
    X = [transform(image=frame)['image'] for frame in X]
    X = np.array(X)
    X = X.transpose((3, 0, 1, 2))
    X = np.expand_dims(X, axis=0)
    # X = torch.from_numpy(X).unsqueeze(0).to(device)
    return np.ascontiguousarray(X.astype(np.float32))



