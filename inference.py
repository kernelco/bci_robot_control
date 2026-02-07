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

from typing import Any

from holoscan.core import (
    ExecutionContext,
   	InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from .window import WindowOutput

class InferenceOperator(Operator):
    def __init__(self, *, fragment: Any | None = None) -> None:
        super().__init__(fragment, name=self.__class__.__name__)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("window")

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del op_output, context

        window: WindowOutput = op_input.receive("window")

        print(f"nirs={window.nirs_window.shape}, eeg={window.eeg_window.shape}")
