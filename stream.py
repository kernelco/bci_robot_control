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

import logging
from typing import Any, NamedTuple

import numpy as np

from streams.base_eeg import BaseEegStream

from streams.base_nirs import BaseNirsStream, ChannelInfo
from holoscan.core import (
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)

logger = logging.getLogger(__name__)


class SampleOutput(NamedTuple):
    data: np.ndarray
    channels: ChannelInfo


class NirsStreamOperator(Operator):
    def __init__(
        self,
        stream: BaseNirsStream,
        *,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._stream = stream
        self._channels: ChannelInfo

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("samples")

    def start(self) -> None:
        self._stream.start()
        self._channels = self._stream.get_channels()
        self._iter = self._stream.stream_nirs()

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del op_input

        sample = next(self._iter, None)
        if sample is None:
            raise StopIteration

        op_output.emit(SampleOutput(sample, self._channels), "samples")


class EegStreamOperator(Operator):
    def __init__(
        self,
        stream: BaseEegStream,
        *,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._stream = stream

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("eeg_data")

    def start(self) -> None:
        self._stream.start()
        self._iter = self._stream.stream_eeg()

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del op_input

        eeg_data = next(self._iter, None)
        if eeg_data is None:
            raise StopIteration

        op_output.emit(eeg_data, "eeg_data")
