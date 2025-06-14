import numpy as np
from numpy.typing import NDArray


class Neuron:
    def __init__(self, bias_neuron: bool = False) -> None:
        # TODO いずれこのweightとbiasは変更する必要がありそう（ローカル最小に引っかかったり学習速度の関係があるから）
        self.bias_neuron = bias_neuron

    def activate(self, activation: float) -> float:
        self.output = self._activation_function(activation)
        return self.output

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # ユニット入力→活性化→活性化関数→ユニット出力の流れを総合して行う
        if self.bias_neuron:
            return float(1.0)

        a = self._make_activations(x)
        y = self._activation_function(a)
        return float(y)

    def backward(
        self,
        delta_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pass

    def _make_activations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # forward()は一つのunitに関してinputからoutputを計算するmethod

        a = np.dot(self.weight, x)
        return a

    def _activation_function(self, a: float) -> float:
        # 今回はtanhを使用します
        y = np.tanh(a)  # (e^x - e^(-x))/(e^x + e^(-x))
        return y
