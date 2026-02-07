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

import abc
from typing import Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray


class ChannelInfo(NamedTuple):
    sds: NDArray[np.float32]
    detector_module: NDArray[np.int_]
    detector_number: NDArray[np.int_]
    source_module: NDArray[np.int_]
    source_number: NDArray[np.int_]

    def __len__(self) -> int:
        return len(self.sds)


class BaseNirsStream(abc.ABC):
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def get_channels(self) -> ChannelInfo:
        pass

    @abc.abstractmethod
    def stream_nirs(self) -> Iterator[np.ndarray]:
        pass

