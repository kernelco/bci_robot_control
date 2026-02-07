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

from typing import Sequence

from streams.kernel_sdk import KernelSDKNirsStream, KernelSdkEegStream
from holoscan.core import Application, Operator
from operators.inference import InferenceOperator
from operators.window import WindowOperator
from operators.stream import EegStreamOperator, NirsStreamOperator


class RobotControlApplication(Application):
	def __init__(
		self,
		*,
		nirs_window_size: int = 100,
		eeg_window_size: int = 100,
	) -> None:
		super().__init__()
		self._nirs_window_size = nirs_window_size
		self._eeg_window_size = eeg_window_size

	def compose(self) -> Sequence[Operator]:
		fragment = self

		nirs_operator = NirsStreamOperator(stream=KernelSDKNirsStream(), fragment=fragment)
		eeg_operator = EegStreamOperator(stream=KernelSdkEegStream(), fragment=fragment)
		window_operator = WindowOperator(
			nirs_window_size=self._nirs_window_size,
			eeg_window_size=self._eeg_window_size,
			fragment=fragment,
		)
		inference_operator = InferenceOperator(fragment=fragment)

		self.add_flow(nirs_operator, window_operator, {("samples", "nirs_samples")})
		self.add_flow(eeg_operator, window_operator, {("eeg_data", "eeg_samples")})
		self.add_flow(window_operator, inference_operator, {("window", "window")})


if __name__ == "__main__":
    nirs_window_size = 72 # 15s @ 4.76Hz
    eeg_window_size = 7499 # 15s @ 500Hz
    app = RobotControlApplication(nirs_window_size=nirs_window_size, eeg_window_size=eeg_window_size)
    app.run()
