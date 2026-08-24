import numpy as np
from neuron import Neuron


class Layer:

    def __init__(self, num_neurons, inputs_size, activation="sigmoid", dropout_rate=0.0):
        self.neurons = [Neuron(inputs_size, activation) for _ in range(num_neurons)]
        self.dropout_rate = dropout_rate  # 0.0 = sin dropout, 0.2 = 20% neuronas apagadas
        self.dropout_mask = None          # máscara de qué neuronas están activas
        self.training = True              # True durante entrenamiento, False en predicción

        for neuron in self.neurons:
            neuron.m_weight = np.zeros_like(neuron.weight)
            neuron.v_weight = np.zeros_like(neuron.weight)
            neuron.m_bias = 0.0
            neuron.v_bias = 0.0

    def forward(self, inputs):
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        # Dropout solo durante entrenamiento
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=outputs.shape)
            outputs = outputs * self.dropout_mask / (1 - self.dropout_rate)
        else:
            self.dropout_mask = np.ones(len(self.neurons))

        return outputs

    def backward(self, d_output):
        # Aplicar la misma máscara de dropout al gradiente
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            d_output = d_output * self.dropout_mask / (1 - self.dropout_rate)

        d_inputs = np.zeros(len(self.neurons[0].inputs))
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_output[i])
        return d_inputs

    def update(self, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for neuron in self.neurons:
            neuron.m_weight = beta1 * neuron.m_weight + (1 - beta1) * neuron.dweight
            neuron.v_weight = beta2 * neuron.v_weight + (1 - beta2) * (neuron.dweight ** 2)

            m_w_corrected = neuron.m_weight / (1 - beta1 ** t)
            v_w_corrected = neuron.v_weight / (1 - beta2 ** t)

            neuron.weight -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)

            neuron.m_bias = beta1 * neuron.m_bias + (1 - beta1) * neuron.dbias
            neuron.v_bias = beta2 * neuron.v_bias + (1 - beta2) * (neuron.dbias ** 2)

            m_b_corrected = neuron.m_bias / (1 - beta1 ** t)
            v_b_corrected = neuron.v_bias / (1 - beta2 ** t)

            neuron.bias -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

            neuron.dweight = np.zeros_like(neuron.weight)
            neuron.dbias = 0.0

    def to_dict(self):
        return [neuron.to_dict() for neuron in self.neurons]

    def from_dict(self, data):
        for neuron, neuron_data in zip(self.neurons, data):
            neuron.from_dict(neuron_data)


if __name__ == "__main__":
    layer = Layer(3, 4, dropout_rate=0.2)
    inputs = np.array([1.0, 8.0, 5.0, 6.0])

    layer.training = True
    layer_output = layer.forward(inputs)
    print("Layer outputs (entrenamiento):", layer_output)

    layer.training = False
    layer_output = layer.forward(inputs)
    print("Layer outputs (predicción):", layer_output)

    d_output = np.array([0.1, -0.2, 0.05])
    d_inputs = layer.backward(d_output)
    print("Gradientes hacia capa anterior:", d_inputs)

    layer.update(learning_rate=0.01, t=1)
    print("Pesos actualizados correctamente con Adam + Dropout")