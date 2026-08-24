import numpy as np


class Neuron:

    def __init__(self, n_input, activation='sigmoid'):
        if activation == 'relu':
            self.weight = np.random.randn(n_input) * np.sqrt(2 / n_input)
        else:
            self.weight = np.random.randn(n_input) * np.sqrt(1 / n_input)

        self.bias = 0.0
        self.output = 0
        self.inputs = None
        self.dweight = np.zeros_like(self.weight)
        self.dbias = 0.0
        self.activation = activation

        self.m_weight = np.zeros_like(self.weight)
        self.v_weight = np.zeros_like(self.weight)
        self.m_bias = 0.0
        self.v_bias = 0.0
        self.t = 0

    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        return 1 / (1 + np.exp(-x))

    def derivate_activate(self, x):
        if self.activation == 'relu':
            return np.where(x > 0, 1.0, 0.0)
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weight) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output

    def backward(self, d_output):
        d_activate = d_output * self.derivate_activate(self.output)
        self.dweight += self.inputs * d_activate
        self.dbias += d_activate
        d_input = np.dot(d_activate, self.weight)
        return d_input

    def update(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        self.m_weight = beta1 * self.m_weight + (1 - beta1) * self.dweight
        self.v_weight = beta2 * self.v_weight + (1 - beta2) * (self.dweight ** 2)
        self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.dbias
        self.v_bias = beta2 * self.v_bias + (1 - beta2) * (self.dbias ** 2)

        m_weight_corr = self.m_weight / (1 - beta1 ** self.t)
        v_weight_corr = self.v_weight / (1 - beta2 ** self.t)
        m_bias_corr = self.m_bias / (1 - beta1 ** self.t)
        v_bias_corr = self.v_bias / (1 - beta2 ** self.t)

        self.weight -= learning_rate * m_weight_corr / (np.sqrt(v_weight_corr) + epsilon)
        self.bias -= learning_rate * m_bias_corr / (np.sqrt(v_bias_corr) + epsilon)

        self.dweight = np.zeros_like(self.weight)
        self.dbias = 0.0

    def to_dict(self):
        return {
            "weights": self.weight.tolist(),
            "bias": float(self.bias)
            if isinstance(self.bias, np.ndarray)
            else self.bias,
        }

    def from_dict(self, data):
        self.weight = np.array(data["weights"])
        self.bias = data["bias"]


if __name__ == "__main__":
    neuron = Neuron(3)

    dataset = [
        (np.array([1.0, 1.0, 1.0]), 1.0),
        (np.array([0.9, 0.8, 0.9]), 1.0),
        (np.array([0.1, 0.1, 0.1]), 0.0),
        (np.array([0.2, 0.1, 0.2]), 0.0),
    ]

    print("=== Entrenamiento con He initialization + Adam ===\n")

    for epoca in range(5000):
        error_total = 0
        for inputs, target in dataset:
            output = neuron.forward(inputs)
            error = -(target - output)
            error_total += (target - output) ** 2
            neuron.backward(d_output=error)
        neuron.update(learning_rate=0.01)
        if epoca % 500 == 0:
            print(f"Época {epoca:3d} | Error total: {error_total:.6f}")

    print("\n=== Resultados finales ===")
    for inputs, target in dataset:
        output = neuron.forward(inputs)
        print(f"Input: {inputs} | Target: {target} | Output: {output:.4f}")