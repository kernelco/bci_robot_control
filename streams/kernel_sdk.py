# Copyright 2026 Kernel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from queue import Empty, Full, Queue
from threading import Thread
from typing import Iterator, List, Tuple, cast

import numpy as np
from kernel.sdk import MomentNumber, SdkClient, Wavelength
from kernel.sdk.socket import (
    Event,
    check_faulted_on_command_failure,
    requires_flow_booted,
    requires_flow_connected,
)

from .base_eeg import BaseEegStream

from . import dist3d
from .base_nirs import BaseNirsStream, ChannelInfo

logger = logging.getLogger(__name__)


class KernelSDKReceiver(SdkClient):
    """
    A class to receive data from the Flow device using the Kernel SDK.
    """

    @property
    @check_faulted_on_command_failure
    @requires_flow_connected
    @requires_flow_booted
    def iter_nirs(self) -> Iterator[np.ndarray]:
        """
        Get an infinite iterator over the flow device's data.

        subscribes to nirs data and yields samples
        """
        yield from self._iter_full_moments()

    @property
    @check_faulted_on_command_failure
    @requires_flow_connected
    @requires_flow_booted
    def iter_eeg(self) -> Iterator[np.ndarray]:
        """
        Get an infinite iterator over the flow device's EEG data.

        subscribes to EEG data and yields samples
        """
        yield from self._iter_eeg()

    def _iter_full_moments(self) -> Iterator[np.ndarray]:
        data_queue: Queue[np.ndarray] = Queue(5)
        moment_ids = [MomentNumber.Zeroth, MomentNumber.First, MomentNumber.Second]
        wavelengths = [Wavelength.Red, Wavelength.IR]

        # need to yield  (num_modules, num_sources, num_modules, num_detectors) shaped data into data_queue

        def multi_moments_callback(data_dict: dict):
            logger.debug("NIRS data callback triggered")

            moment_slices = []
            for moment_id in moment_ids:
                wavelength_slices = []
                for wavelength in wavelengths:
                    urn = self.MOMENTS_FIELD_URN_TEMPLATE.format(
                        moment_id=moment_id.value, wavelength=wavelength.value
                    )
                    field_data = data_dict.get(urn)
                    if field_data is None:
                        print(
                            f"Missing data for {urn} among {list(data_dict.keys())}, skipping callback."
                        )
                        return
                    wavelength_slices.append(field_data)
                moment_slices.append(np.stack(wavelength_slices, axis=-1))
            stacked_data = np.stack(moment_slices, axis=-2)[:, :, :, :, :6, :, :]

            for sample in stacked_data:
                try:
                    data_queue.put_nowait(sample)
                except Full:
                    # this happens when the pipeline is not consuming fast enough
                    # mainly during cache/warmup phase
                    logger.debug("Data queue is full. Dropping sample.")

        # Subscribe to all moment_id and wavelength combinations
        urns = [
            self.MOMENTS_FIELD_URN_TEMPLATE.format(
                moment_id=moment_id.value, wavelength=wavelength.value
            )
            for moment_id in moment_ids
            for wavelength in wavelengths
        ]
        logger.info(f"Subscribing to NIRS data URNs: {urns}")
        nirs_future = self._sdk.new_event(
            Event(device="flow", field_urns=urns), multi_moments_callback
        )

        try:
            while True:
                # Block until data is available in the queue
                logger.debug("Waiting for NIRS data...")
                yield data_queue.get()
                logger.debug("NIRS data received. queue size: %d", data_queue.qsize())
        finally:
            nirs_future.cancel()

    def _iter_eeg(self) -> Iterator[np.ndarray]:
        data_queue: Queue[np.ndarray] = Queue(5)

        def eeg_callback(timestamp, data):
            # data is (n_samples, num_channels=8, voltage/impedance=2)
            try:
                data_queue.put_nowait(data)
            except Full:
                # this happens when the pipeline is not consuming fast enough
                # mainly during cache/warmup phase
                logger.debug("Data queue is full. Dropping sample.")

        logger.info(f"Subscribing to EEG data")
        eeg_future = self.on_eeg(eeg_callback)

        try:
            while True:
                # Block until data is available in the queue
                logger.debug("Waiting for EEG data...")
                yield data_queue.get()
                logger.debug("EEG data received. queue size: %d", data_queue.qsize())
        finally:
            eeg_future.cancel()


class KernelSDKNirsStream(BaseNirsStream):
    MAX_MODULE_COUNT = 48
    NUM_SOURCES = 3
    NUM_DETECTORS = 6

    def __init__(
        self,
        *,
        receiver_queue_size: int = 32,
    ) -> None:
        if receiver_queue_size <= 0:
            raise ValueError("receiver_queue_size must be positive")
        self._receiver_queue_size = int(receiver_queue_size)
        self._receiver_queue: Queue[np.ndarray] | None = None
        self._receiver_thread: Thread | None = None
        self._channels: ChannelInfo | None = None
        self._good_channels: List[int] = []

    def start(self) -> None:
        self._receiver = KernelSDKReceiver()
        if self._receiver_queue is None:
            self._receiver_queue = Queue(self._receiver_queue_size)
        self._receiver_thread = Thread(
            target=self._receiver_loop,
            name="KernelSDKReceiverThread",
            daemon=True,
        )
        self._receiver_thread.start()
        self._channels = self._build_all_channels()

    def _build_all_channels(self) -> ChannelInfo:
        nirs_data: np.ndarray
        if self._receiver_queue is None:
            raise RuntimeError("KernelSDKStream.start() must be called before streaming NIRS data")
        while True:
            try:
                # nirs_data is (num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths) = (48, 3, 48, 6, 3, 2)
                nirs_data = self._receiver_queue.get(timeout=1.0)
                logger.debug("Received new NIRS data frame for channel building")
                break
            except Empty:
                continue

        sds = []
        detector_module = []
        detector_number = []
        source_module = []
        source_number = []

        channel_idx = -1
        for source_module_id in range(self.MAX_MODULE_COUNT):
            for source_num in range(self.NUM_SOURCES):
                for detector_module_id in range(self.MAX_MODULE_COUNT):
                    for detector_num in range(self.NUM_DETECTORS):
                        channel_idx += 1

                        # if there are any nan values for this channel, skip it
                        channel_data = nirs_data[
                            source_module_id, source_num, detector_module_id, detector_num
                        ]
                        # if any are not finite, skip
                        if not np.isfinite(channel_data).all():
                            continue

                        self._good_channels.append(channel_idx)

                        detector_module.append(detector_module_id)
                        detector_number.append(detector_num)
                        source_module.append(source_module_id)
                        source_number.append(source_num)

                        x1, y1, z1 = cast(
                            Tuple[float, float, float],
                            LOCATIONS["flow2"]["source"][source_module_id][0][source_num][0],
                        )
                        x2, y2, z2 = cast(
                            Tuple[float, float, float],
                            LOCATIONS["flow2"]["detector"][detector_module_id][0][detector_num][0],
                        )
                        dist = dist3d(x1, y1, z1, x2, y2, z2)
                        sds.append(dist)

        return ChannelInfo(
            np.array(source_module),
            np.array(source_number),
            np.array(detector_module),
            np.array(detector_number),
            np.array(sds),
        )

    def _receiver_loop(self):
        """
        Dedicated thread that receives data from the KernelSDKReceiver and puts it into a queue.
        Prevents internal queue filling up when stream_nirs is not consuming fast enough.
        """
        if self._receiver_queue is None:
            return
        try:
            for frame in self._receiver.iter_nirs:
                queue = self._receiver_queue
                if queue is None:
                    break
                try:
                    queue.put_nowait(frame)
                except Full:
                    # this happens when the pipeline is not consuming fast enough
                    # mainly during cache/warmup phase
                    logger.debug("KernelSDKStream receiver queue full; dropping frame")
        except Exception:
            logger.exception("KernelSDKStream receiver loop crashed")

    def get_channels(self) -> ChannelInfo:
        if self._channels is None:
            raise RuntimeError("KernelSDKStream.start() must be called before getting channels")
        return self._channels

    def stream_nirs(self):
        if self._receiver_queue is None:
            raise RuntimeError("KernelSDKStream.start() must be called before streaming NIRS data")
        while True:
            try:
                # nirs_data is (num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths) = (48, 3, 48, 6, 3, 2)
                nirs_data = self._receiver_queue.get(timeout=1.0)
            except Empty:
                continue
            logger.debug("Received new NIRS data frame")

            # flatten first 4 dimensions into one dimension
            nirs_data = nirs_data.reshape(-1, nirs_data.shape[4], nirs_data.shape[5])

            # channels, moments, wavelengths => moments, channels, wavelengths
            reshaped = nirs_data.transpose(1, 0, 2)
            logger.debug("NIRS data shape after reshaping: %s", reshaped.shape)
            reshaped = reshaped[:, self._good_channels, :]
            logger.debug("NIRS data shape after channel filtering: %s", reshaped.shape)

            yield reshaped


class KernelSdkEegStream(BaseEegStream):
    def __init__(
        self,
        *,
        receiver_queue_size: int = 32,
    ) -> None:
        if receiver_queue_size <= 0:
            raise ValueError("receiver_queue_size must be positive")
        self._receiver_queue_size = int(receiver_queue_size)
        self._receiver_queue: Queue[np.ndarray] | None = None
        self._receiver_thread: Thread | None = None

    def start(self) -> None:
        self._receiver = KernelSDKReceiver()
        if self._receiver_queue is None:
            self._receiver_queue = Queue(self._receiver_queue_size)
        self._receiver_thread = Thread(
            target=self._receiver_loop,
            name="KernelSDKEEGReceiverThread",
            daemon=True,
        )
        self._receiver_thread.start()

    def _receiver_loop(self):
        """
        Dedicated thread that receives data from the KernelSDKReceiver and puts it into a queue.
        Prevents internal queue filling up when stream_eeg is not consuming fast enough.
        """
        if self._receiver_queue is None:
            return
        try:
            for frame in self._receiver.iter_eeg:
                queue = self._receiver_queue
                if queue is None:
                    break
                try:
                    queue.put_nowait(frame)
                except Full:
                    # this happens when the pipeline is not consuming fast enough
                    # mainly during cache/warmup phase
                    logger.debug("KernelSdkEegStream receiver queue full; dropping frame")
        except Exception:
            logger.exception("KernelSdkEegStream receiver loop crashed")

    def stream_eeg(self) -> Iterator[np.ndarray]:
        if self._receiver_queue is None:
            raise RuntimeError("KernelSdkEegStream.start() must be called before streaming EEG data")
        while True:
            try:
                eeg_data = self._receiver_queue.get(timeout=1.0)
                yield eeg_data
            except Empty:
                continue

LOCATIONS = {
    "flow2": {
        "source": [
            [[[[11.691, 91.053, 0.648]], [[-11.691, 91.091, 0.648]], [[0.0, 88.854, 20.934]]]],
            [[[[-11.691, 80.465, 43.408]], [[0.0, 70.263, 60.925]], [[11.691, 80.59, 43.489]]]],
            [[[[46.141, 76.113, -1.445]], [[26.206, 88.352, -1.125]], [[35.603, 80.67, 18.964]]]],
            [[[[-26.142, 88.24, -1.125]], [[-46.144, 76.117, -1.445]], [[-35.53, 80.543, 18.964]]]],
            [[[[46.804, 57.831, 46.527]], [[28.062, 71.651, 48.746]], [[31.937, 54.627, 64.299]]]],
            [
                [
                    [[-46.83, 58.073, 46.449]],
                    [[-31.949, 54.851, 64.206]],
                    [[-27.932, 71.696, 48.492]],
                ]
            ],
            [[[[71.05, 46.887, -6.725]], [[62.354, 55.425, 13.26]], [[73.986, 35.141, 13.308]]]],
            [
                [
                    [[-71.05, 47.144, -6.711]],
                    [[-73.686, 35.226, 13.283]],
                    [[-62.289, 55.645, 13.265]],
                ]
            ],
            [[[[61.0, 31.519, 54.722]], [[74.346, 22.76, 37.606]], [[64.571, 43.93, 35.219]]]],
            [[[[-61.196, 31.381, 54.799]], [[-64.845, 43.841, 35.339]], [[-74.3, 22.472, 37.554]]]],
            [[[[5.483, 34.477, 88.606]], [[24.236, 20.518, 90.787]], [[25.874, 40.625, 78.907]]]],
            [
                [
                    [[-5.461, 34.268, 88.693]],
                    [[-25.905, 40.566, 79.228]],
                    [[-24.242, 20.387, 90.995]],
                ]
            ],
            [[[[54.207, 16.684, 71.524]], [[36.368, 14.358, 86.724]], [[50.966, -3.561, 82.832]]]],
            [
                [
                    [[-54.115, 16.878, 71.608]],
                    [[-50.799, -3.403, 82.842]],
                    [[-36.243, 14.535, 86.776]],
                ]
            ],
            [[[[84.236, 5.385, 3.042]], [[78.6, 20.77, 19.739]], [[83.511, -1.383, 25.413]]]],
            [[[[-83.967, 5.547, 2.968]], [[-83.442, -1.176, 25.357]], [[-78.177, 20.899, 19.65]]]],
            [[[[5.101, -3.374, 103.879]], [[25.58, -14.514, 101.035]], [[23.819, 8.152, 95.565]]]],
            [
                [
                    [[-5.122, -3.585, 103.954]],
                    [[-23.83, 7.927, 95.581]],
                    [[-25.622, -14.692, 101.244]],
                ]
            ],
            [[[[72.096, 3.904, 57.069]], [[76.942, -18.975, 57.026]], [[82.259, -5.882, 38.382]]]],
            [[[[-72.029, 3.776, 57.239]], [[-82.237, -6.0, 38.573]], [[-76.802, -19.121, 57.159]]]],
            [
                [
                    [[59.752, -24.442, 78.661]],
                    [[42.007, -18.898, 92.922]],
                    [[47.409, -41.369, 89.046]],
                ]
            ],
            [
                [
                    [[-59.764, -24.231, 78.768]],
                    [[-47.481, -41.162, 89.22]],
                    [[-42.047, -18.689, 93.061]],
                ]
            ],
            [
                [
                    [[17.589, -54.933, 101.982]],
                    [[30.651, -35.559, 100.611]],
                    [[8.404, -34.182, 107.679]],
                ]
            ],
            [
                [
                    [[-17.803, -54.947, 101.92]],
                    [[-8.628, -34.203, 107.653]],
                    [[-30.917, -35.606, 100.724]],
                ]
            ],
            [[[[87.248, -35.431, 11.589]], [[85.604, -21.07, 29.971]], [[83.286, -44.15, 32.928]]]],
            [
                [
                    [[-87.296, -35.213, 11.564]],
                    [[-83.592, -43.951, 32.936]],
                    [[-85.611, -20.848, 29.941]],
                ]
            ],
            [
                [
                    [[71.244, -64.382, 52.672]],
                    [[79.172, -42.649, 48.86]],
                    [[67.475, -47.658, 68.575]],
                ]
            ],
            [
                [
                    [[-71.231, -64.29, 52.406]],
                    [[-67.581, -47.608, 68.383]],
                    [[-79.384, -42.637, 48.735]],
                ]
            ],
            [
                [
                    [[46.093, -76.176, 75.331]],
                    [[25.316, -80.528, 85.228]],
                    [[32.523, -93.916, 67.239]],
                ]
            ],
            [
                [
                    [[-46.057, -76.08, 75.549]],
                    [[-32.395, -93.733, 67.327]],
                    [[-25.152, -80.31, 85.265]],
                ]
            ],
            [
                [
                    [[41.723, -101.815, 46.602]],
                    [[21.224, -103.86, 58.28]],
                    [[23.878, -113.429, 36.904]],
                ]
            ],
            [
                [
                    [[-41.753, -101.833, 46.844]],
                    [[-23.626, -112.895, 36.954]],
                    [[-20.963, -103.311, 58.324]],
                ]
            ],
            [
                [
                    [[74.391, -77.978, 21.37]],
                    [[61.573, -88.435, 37.963]],
                    [[61.434, -96.698, 16.038]],
                ]
            ],
            [
                [
                    [[-74.24, -77.786, 21.551]],
                    [[-61.329, -96.539, 16.227]],
                    [[-61.51, -88.308, 38.16]],
                ]
            ],
            [
                [
                    [[72.605, -84.782, -16.612]],
                    [[62.702, -97.731, 0.157]],
                    [[56.508, -100.83, -22.239]],
                ]
            ],
            [
                [
                    [[-72.391, -84.512, -16.383]],
                    [[-56.446, -100.695, -22.019]],
                    [[-62.555, -97.519, 0.382]],
                ]
            ],
            [
                [
                    [[27.558, -118.923, 12.288]],
                    [[43.379, -112.78, -3.811]],
                    [[47.442, -108.366, 18.967]],
                ]
            ],
            [
                [
                    [[-27.332, -118.518, 12.078]],
                    [[-47.488, -108.611, 18.747]],
                    [[-43.354, -112.854, -4.029]],
                ]
            ],
            [[[[0.0, -104.615, 60.754]], [[11.691, -92.436, 77.13]], [[-11.691, -92.172, 76.877]]]],
            [
                [
                    [[0.0, -121.502, 17.483]],
                    [[11.691, -115.523, 37.099]],
                    [[-11.691, -115.07, 37.036]],
                ]
            ],
        ],
        "detector": [
            [
                [
                    [[5.275, 92.111, -5.045]],
                    [[-5.275, 92.126, -5.045]],
                    [[-13.399, 89.855, 9.063]],
                    [[-8.125, 89.007, 18.213]],
                    [[8.125, 89.002, 18.213]],
                    [[13.399, 89.821, 9.063]],
                    [[0.0, 91.048, 7.399]],
                ]
            ],
            [
                [
                    [[-13.399, 76.122, 50.597]],
                    [[-8.125, 71.527, 58.506]],
                    [[8.125, 71.672, 58.599]],
                    [[13.399, 76.294, 50.708]],
                    [[5.275, 83.526, 38.627]],
                    [[-5.275, 83.471, 38.591]],
                    [[0.0, 78.145, 49.944]],
                ]
            ],
            [
                [
                    [[41.457, 81.007, -7.034]],
                    [[32.508, 86.609, -6.89]],
                    [[24.123, 87.914, 7.294]],
                    [[28.677, 85.0, 16.357]],
                    [[42.29, 76.074, 16.135]],
                    [[46.913, 73.785, 6.927]],
                    [[36.513, 82.641, 5.464]],
                ]
            ],
            [
                [
                    [[-32.378, 86.381, -6.889]],
                    [[-41.411, 80.926, -7.034]],
                    [[-46.915, 73.788, 6.927]],
                    [[-42.289, 76.071, 16.135]],
                    [[-28.557, 84.79, 16.357]],
                    [[-24.077, 87.834, 7.294]],
                    [[-36.432, 82.498, 5.464]],
                ]
            ],
            [
                [
                    [[43.074, 64.296, 42.318]],
                    [[34.655, 70.579, 43.363]],
                    [[24.435, 68.495, 55.84]],
                    [[26.092, 60.698, 62.754]],
                    [[39.375, 51.421, 61.504]],
                    [[46.059, 52.837, 53.459]],
                    [[36.031, 61.916, 53.68]],
                ]
            ],
            [
                [
                    [[-46.14, 53.149, 53.444]],
                    [[-39.44, 51.713, 61.472]],
                    [[-26.043, 60.844, 62.59]],
                    [[-24.299, 68.531, 55.579]],
                    [[-34.594, 70.712, 43.187]],
                    [[-43.05, 64.476, 42.185]],
                    [[-36.018, 62.109, 53.558]],
                ]
            ],
            [
                [
                    [[66.598, 53.706, -4.058]],
                    [[62.422, 57.413, 4.926]],
                    [[65.215, 49.682, 18.943]],
                    [[70.455, 40.526, 18.963]],
                    [[75.648, 34.106, 4.967]],
                    [[74.435, 39.469, -4.057]],
                    [[69.562, 46.065, 6.67]],
                ]
            ],
            [
                [
                    [[-74.274, 39.634, -4.064]],
                    [[-75.317, 34.173, 4.938]],
                    [[-70.308, 40.699, 18.958]],
                    [[-65.212, 49.938, 18.956]],
                    [[-62.317, 57.61, 4.926]],
                    [[-66.577, 53.951, -4.047]],
                    [[-69.469, 46.269, 6.671]],
                ]
            ],
            [
                [
                    [[65.883, 24.627, 53.282]],
                    [[72.071, 20.779, 45.649]],
                    [[73.523, 28.763, 31.544]],
                    [[69.219, 38.381, 30.523]],
                    [[60.425, 44.766, 42.675]],
                    [[58.651, 39.065, 51.387]],
                    [[67.426, 33.227, 42.939]],
                ]
            ],
            [
                [
                    [[-58.945, 38.988, 51.518]],
                    [[-60.757, 44.713, 42.826]],
                    [[-69.33, 38.19, 30.556]],
                    [[-73.463, 28.466, 31.484]],
                    [[-72.096, 20.534, 45.636]],
                    [[-65.99, 24.433, 53.312]],
                    [[-67.56, 33.05, 42.983]],
                ]
            ],
            [
                [
                    [[7.624, 27.203, 92.598]],
                    [[16.074, 20.874, 93.533]],
                    [[30.068, 24.641, 86.007]],
                    [[30.79, 33.666, 80.574]],
                    [[18.006, 44.075, 80.162]],
                    [[8.747, 41.132, 84.274]],
                    [[18.777, 32.577, 87.195]],
                ]
            ],
            [
                [
                    [[-8.72, 40.908, 84.337]],
                    [[-17.998, 43.904, 80.307]],
                    [[-30.848, 33.685, 81.015]],
                    [[-30.098, 24.579, 86.322]],
                    [[-16.074, 20.726, 93.716]],
                    [[-7.609, 27.012, 92.712]],
                    [[-18.797, 32.488, 87.468]],
                ]
            ],
            [
                [
                    [[47.865, 21.464, 74.798]],
                    [[39.81, 20.411, 81.649]],
                    [[37.372, 6.254, 89.325]],
                    [[44.063, -1.78, 87.672]],
                    [[56.297, -0.24, 76.95]],
                    [[57.711, 8.871, 71.802]],
                    [[47.623, 9.369, 80.791]],
                ]
            ],
            [
                [
                    [[-57.589, 9.05, 71.856]],
                    [[-56.124, -0.085, 76.955]],
                    [[-43.899, -1.621, 87.684]],
                    [[-37.184, 6.402, 89.315]],
                    [[-39.757, 20.622, 81.77]],
                    [[-47.807, 21.673, 74.914]],
                    [[-47.479, 9.538, 80.823]],
                ]
            ],
            [
                [
                    [[82.404, 13.726, 3.721]],
                    [[79.725, 20.638, 11.241]],
                    [[79.087, 15.906, 26.775]],
                    [[81.37, 5.926, 29.341]],
                    [[84.917, -4.847, 17.705]],
                    [[85.15, -1.663, 7.726]],
                    [[82.435, 8.374, 16.28]],
                ]
            ],
            [
                [
                    [[-85.061, -1.612, 7.545]],
                    [[-84.867, -4.636, 17.65]],
                    [[-81.21, 6.112, 29.277]],
                    [[-78.763, 16.056, 26.695]],
                    [[-79.252, 20.756, 11.148]],
                    [[-81.995, 13.858, 3.633]],
                    [[-82.17, 8.491, 16.02]],
                ]
            ],
            [
                [
                    [[8.374, -11.179, 105.215]],
                    [[17.647, -16.153, 104.139]],
                    [[30.505, -8.372, 97.436]],
                    [[29.713, 1.859, 94.985]],
                    [[15.596, 9.776, 97.673]],
                    [[7.136, 4.551, 101.325]],
                    [[18.288, -3.06, 100.913]],
                ]
            ],
            [
                [
                    [[-7.158, 4.342, 101.41]],
                    [[-15.612, 9.558, 97.719]],
                    [[-29.725, 1.635, 95.008]],
                    [[-30.537, -8.564, 97.584]],
                    [[-17.684, -16.339, 104.315]],
                    [[-8.397, -11.385, 105.306]],
                    [[-18.313, -3.264, 101.015]],
                ]
            ],
            [
                [
                    [[70.846, -2.934, 62.087]],
                    [[73.166, -13.226, 62.133]],
                    [[80.594, -19.94, 49.328]],
                    [[82.926, -14.048, 40.883]],
                    [[79.828, 1.915, 41.047]],
                    [[75.441, 6.376, 49.577]],
                    [[78.141, -6.74, 51.343]],
                ]
            ],
            [
                [
                    [[-75.39, 6.252, 49.756]],
                    [[-79.796, 1.794, 41.235]],
                    [[-82.86, -14.177, 41.053]],
                    [[-80.488, -20.078, 49.479]],
                    [[-72.999, -13.378, 62.253]],
                    [[-70.712, -3.078, 62.223]],
                    [[-78.04, -6.877, 51.496]],
                ]
            ],
            [
                [
                    [[55.762, -17.393, 81.558]],
                    [[47.78, -14.893, 88.019]],
                    [[39.42, -26.668, 95.478]],
                    [[41.829, -36.805, 93.697]],
                    [[54.114, -40.655, 83.735]],
                    [[59.707, -33.019, 79.076]],
                    [[50.204, -28.262, 87.409]],
                ]
            ],
            [
                [
                    [[-59.749, -32.81, 79.217]],
                    [[-54.167, -40.446, 83.888]],
                    [[-41.919, -36.598, 93.891]],
                    [[-39.505, -26.461, 95.666]],
                    [[-47.763, -14.681, 88.095]],
                    [[-55.731, -17.18, 81.618]],
                    [[-50.255, -28.053, 87.56]],
                ]
            ],
            [
                [
                    [[25.598, -52.73, 99.88]],
                    [[31.461, -43.97, 99.161]],
                    [[25.103, -29.565, 103.202]],
                    [[15.076, -28.951, 106.427]],
                    [[5.965, -42.392, 107.263]],
                    [[10.101, -51.75, 104.667]],
                    [[19.294, -41.817, 104.816]],
                ]
            ],
            [
                [
                    [[-10.295, -51.751, 104.537]],
                    [[-6.175, -42.404, 107.19]],
                    [[-15.331, -28.991, 106.504]],
                    [[-25.373, -29.615, 103.331]],
                    [[-31.71, -44.007, 99.217]],
                    [[-25.849, -52.768, 99.942]],
                    [[-19.547, -41.856, 104.887]],
                ]
            ],
            [
                [
                    [[87.465, -27.017, 13.2]],
                    [[86.592, -20.528, 21.477]],
                    [[84.453, -26.632, 36.385]],
                    [[83.334, -37.04, 37.709]],
                    [[84.129, -46.994, 24.89]],
                    [[86.003, -43.067, 15.274]],
                    [[85.966, -33.594, 24.905]],
                ]
            ],
            [
                [
                    [[-86.18, -42.858, 15.266]],
                    [[-84.434, -46.795, 24.899]],
                    [[-83.554, -36.834, 37.707]],
                    [[-84.522, -26.416, 36.363]],
                    [[-86.587, -20.306, 21.445]],
                    [[-87.446, -26.794, 13.167]],
                    [[-86.111, -33.383, 24.893]],
                ]
            ],
            [
                [
                    [[75.551, -60.048, 46.611]],
                    [[79.029, -50.207, 44.829]],
                    [[76.678, -38.68, 56.048]],
                    [[71.299, -40.904, 64.879]],
                    [[65.789, -56.006, 67.527]],
                    [[67.432, -63.532, 60.316]],
                    [[73.485, -51.864, 57.237]],
                ]
            ],
            [
                [
                    [[-67.38, -63.426, 60.026]],
                    [[-65.805, -55.925, 67.279]],
                    [[-71.448, -40.869, 64.714]],
                    [[-76.844, -38.651, 55.894]],
                    [[-79.262, -50.202, 44.717]],
                    [[-75.667, -60.001, 46.425]],
                    [[-73.591, -51.815, 57.046]],
                ]
            ],
            [
                [
                    [[41.293, -73.016, 81.716]],
                    [[31.999, -75.055, 86.295]],
                    [[22.27, -87.1, 80.248]],
                    [[25.493, -93.115, 72.093]],
                    [[40.341, -90.477, 65.794]],
                    [[46.383, -82.397, 69.33]],
                    [[35.239, -84.101, 76.773]],
                ]
            ],
            [
                [
                    [[-46.387, -82.338, 69.604]],
                    [[-40.35, -90.423, 66.075]],
                    [[-25.25, -92.822, 72.017]],
                    [[-22.023, -86.805, 80.169]],
                    [[-31.957, -74.952, 86.504]],
                    [[-41.196, -73.098, 81.898]],
                    [[-35.198, -84.0, 76.985]],
                ]
            ],
            [
                [
                    [[38.331, -99.61, 54.233]],
                    [[29.14, -100.645, 59.541]],
                    [[16.385, -108.025, 52.56]],
                    [[17.599, -112.374, 42.927]],
                    [[32.128, -111.505, 35.005]],
                    [[40.103, -106.118, 39.329]],
                    [[29.364, -107.191, 47.549]],
                ]
            ],
            [
                [
                    [[-40.105, -106.081, 39.552]],
                    [[-31.978, -111.172, 35.125]],
                    [[-17.332, -111.813, 42.968]],
                    [[-16.122, -107.471, 52.603]],
                    [[-28.961, -100.254, 59.64]],
                    [[-38.31, -99.528, 54.44]],
                    [[-29.174, -106.779, 47.641]],
                ]
            ],
            [
                [
                    [[72.818, -77.146, 29.765]],
                    [[67.172, -81.966, 37.277]],
                    [[57.706, -94.63, 33.484]],
                    [[57.645, -98.36, 23.593]],
                    [[67.057, -91.47, 12.157]],
                    [[72.86, -82.992, 14.555]],
                    [[66.413, -88.162, 25.24]],
                ]
            ],
            [
                [
                    [[-72.631, -82.741, 14.72]],
                    [[-66.803, -91.201, 12.318]],
                    [[-57.68, -98.306, 23.808]],
                    [[-57.764, -94.594, 33.704]],
                    [[-66.992, -81.752, 37.452]],
                    [[-72.672, -76.957, 29.946]],
                    [[-66.209, -87.93, 25.41]],
                ]
            ],
            [
                [
                    [[72.862, -85.519, -8.074]],
                    [[68.515, -91.468, -0.515]],
                    [[57.364, -102.651, -4.427]],
                    [[54.401, -103.901, -14.523]],
                    [[61.891, -95.439, -26.209]],
                    [[69.186, -88.227, -23.672]],
                    [[64.553, -94.988, -12.932]],
                ]
            ],
            [
                [
                    [[-68.833, -87.834, -23.436]],
                    [[-61.581, -95.085, -25.975]],
                    [[-54.462, -103.874, -14.309]],
                    [[-57.382, -102.585, -4.211]],
                    [[-68.259, -91.161, -0.284]],
                    [[-72.661, -85.261, -7.846]],
                    [[-64.309, -94.692, -12.701]],
                ]
            ],
            [
                [
                    [[28.493, -119.217, 3.743]],
                    [[35.618, -116.414, -3.52]],
                    [[49.494, -109.218, 1.119]],
                    [[51.27, -107.088, 11.399]],
                    [[40.424, -111.712, 22.581]],
                    [[31.413, -116.379, 19.569]],
                    [[39.951, -114.531, 9.129]],
                ]
            ],
            [
                [
                    [[-31.24, -116.101, 19.357]],
                    [[-40.404, -111.797, 22.362]],
                    [[-51.305, -107.306, 11.178]],
                    [[-49.512, -109.394, 0.9]],
                    [[-35.487, -116.236, -3.734]],
                    [[-28.271, -118.821, 3.533]],
                    [[-39.905, -114.554, 8.912]],
                ]
            ],
            [
                [
                    [[8.125, -103.043, 63.014]],
                    [[13.399, -97.495, 70.35]],
                    [[5.275, -89.013, 81.712]],
                    [[-5.275, -88.878, 81.582]],
                    [[-13.399, -97.189, 70.057]],
                    [[-8.125, -102.814, 62.794]],
                    [[0.0, -97.002, 72.156]],
                ]
            ],
            [
                [
                    [[8.125, -120.829, 20.134]],
                    [[13.399, -118.068, 28.976]],
                    [[5.275, -113.784, 42.589]],
                    [[-5.275, -113.552, 42.557]],
                    [[-13.399, -117.574, 28.907]],
                    [[-8.125, -120.493, 20.088]],
                    [[0.0, -118.176, 30.651]],
                ]
            ],
        ],
    }
}

