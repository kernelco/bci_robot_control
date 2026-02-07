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

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional

import numpy as np

from streams.base_nirs import ChannelInfo

from .stream import SampleOutput

from holoscan.core import (
    ConditionType,
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)

class WindowOutput(NamedTuple):
    nirs_window: np.ndarray
    eeg_window: np.ndarray


NUM_MODULES = 40
SDS_BUCKETS = [
    (0, 10),
    (10, 25),
    (25, 60)
]


class WindowOperator(Operator):
    def __init__(
        self,
        *,
        nirs_window_size: int = 100,
        eeg_window_size: int = 100,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        if nirs_window_size <= 0:
            raise ValueError("nirs_window_size must be positive")
        if eeg_window_size <= 0:
            raise ValueError("eeg_window_size must be positive")
        self._nirs_window_size = int(nirs_window_size)
        self._eeg_window_size = int(eeg_window_size)
        self._nirs_window: Deque[np.ndarray] = deque(maxlen=self._nirs_window_size)
        self._eeg_window: Deque[np.ndarray] = deque(maxlen=self._eeg_window_size)
        self._channel_map: Dict[int, Dict[int, List[int]]] = {} # module -> sds_bucket -> list of channel indices

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("nirs_samples").condition(ConditionType.NONE)
        spec.input("eeg_samples").condition(ConditionType.NONE)
        spec.output("window")

    def _build_channel_map(self, channels: ChannelInfo) -> None:
        for channel_idx in range(len(channels)):
            module = channels.source_module[channel_idx]
            sds = channels.sds[channel_idx]
            if module not in self._channel_map:
                self._channel_map[module] = {bucket_idx: [] for bucket_idx in range(len(SDS_BUCKETS))}
            for bucket_idx, bucket in enumerate(SDS_BUCKETS):
                if bucket[0] < sds <= bucket[1]:
                    self._channel_map[module][bucket_idx].append(channel_idx)
                    break

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del context

        incoming_nirs: Optional[SampleOutput] = op_input.receive("nirs_samples")
        if incoming_nirs is not None:
            if not self._channel_map:
                self._build_channel_map(incoming_nirs.channels)
                
            incoming_data = incoming_nirs.data # (moments, channels, wavelengths)

            # take log10 of 0th moment to convert to optical density
            incoming_data[0, :, :] = np.log10(incoming_data[0, :, :])
            
            data = np.full((NUM_MODULES, len(SDS_BUCKETS), incoming_data.shape[2], incoming_data.shape[0]), np.nan) # (modules, sds_buckets, wavelengths, moments)
            for module, bucket_map in self._channel_map.items():
                for bucket_idx, channel_indices in bucket_map.items():
                    if not channel_indices:
                        continue

                    data[module, bucket_idx, :, :] = np.mean(
                        incoming_data[:, channel_indices, :], axis=1
                    ).T # (wavelengths, moments)

            # if any NaNs in data, fill with prior value in window or 0 if no prior value
            if self._nirs_window:
                prior_data = self._nirs_window[-1]
                data = np.where(np.isnan(data), prior_data, data)

            self._nirs_window.append(data)

        incoming_eeg: Optional[np.ndarray] = op_input.receive("eeg_samples")
        if incoming_eeg is not None:
            self._eeg_window.extend(incoming_eeg[:, :, 0]) # (n_samples, n_channels, voltage/impedance) -> just voltage (n_samples, n_channels)

        if (
            len(self._nirs_window) == self._nirs_window_size
            and len(self._eeg_window) == self._eeg_window_size
        ):
            op_output.emit(
                WindowOutput(
                    nirs_window=np.stack(self._nirs_window, axis=0),
                    eeg_window=np.stack(self._eeg_window, axis=0),
                ),
                "window",
            )
