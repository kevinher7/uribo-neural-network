import numpy as np


class Neuron:
    def __init__(self, input_dim, output_dim):
        # TODO いずれこのweightとbiasは偏向する必要がありそう（ローカル最小に引っかかったり学習速度の関係があるから）
        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim)

    def _make_activations(self, x):
        # forward()は一つのunitに関してinputからoutputを計算するmethod
        a = np.dot(self.weight, x) + self.bias
        return a

    def _activation_function(self, a):
        # 今回はtanhを使用します
        y = np.tanh(a)  # (e^x - e^(-x))/(e^x + e^(-x))
        return y

    def forward(self, x):
        # ユニット入力→活性化→活性化関数→ユニット出力の流れを総合して行う
        a = self._make_activations(x)
        y = self._activation_function(a)
        return y
