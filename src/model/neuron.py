import numpy as np
from numpy.typing import NDArray


class Neuron:
    def __init__(self, input_dim: int, output_dim=1) -> None:
        # TODO いずれこのweightとbiasは変更する必要がありそう（ローカル最小に引っかかったり学習速度の関係があるから）
        # TODO 重みベクトルはベクトルに限らず行列になる可能性（output_dim=>2のとき）があるため、それは後で検討する
        self.weight = np.random.randn(input_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # ユニット入力→活性化→活性化関数→ユニット出力の流れを総合して行う
        a = self._make_activations(x)
        y = self._activation_function(a)
        return float(y)

    def _make_activations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # forward()は一つのunitに関してinputからoutputを計算するmethod
        a = np.dot(self.weight, x) + self.bias
        return a

    def _activation_function(self, a: float) -> float:
        # 今回はtanhを使用します
        y = np.tanh(a)  # (e^x - e^(-x))/(e^x + e^(-x))
        return y
