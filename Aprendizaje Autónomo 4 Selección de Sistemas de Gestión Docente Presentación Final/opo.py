import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    """
    Red Neuronal Adaptativa:
    - Para pocos datos: arquitectura simple (ReLU + Dropout)
    - Para muchos datos: arquitectura moderna (BatchNorm + GELU + Weight Decay)
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers_list = nn.ModuleList()
        self.loss_list = []
        self.val_loss_list = []

    def add_layer(self, num_neurons, inputs_size=None, activation="relu", 
                  dropout_rate=0.0, use_batchnorm=False):
        """
        Agrega una capa a la red neuronal.
        
        Args:
            num_neurons: Número de neuronas en la capa
            inputs_size: Tamaño de entrada (requerido solo para la primera capa)
            activation: 'relu', 'gelu', 'sigmoid', 'tanh', 'silu', 'leaky_relu'
            dropout_rate: Tasa de dropout (0.0 = sin dropout)
            use_batchnorm: Si usar BatchNorm1d (False por defecto, mejor para pocos datos)
        """
        # Inferir inputs_size si no se proporciona
        if inputs_size is None:
            if len(self.layers_list) == 0:
                raise ValueError("inputs_size es requerido para la primera capa")
            last_linear = self._get_last_linear()
            if last_linear is None:
                raise ValueError("No se pudo inferir inputs_size de la capa anterior")
            inputs_size = last_linear.out_features
        
        block = []
        
        # 1. Capa Lineal
        linear = nn.Linear(inputs_size, num_neurons)
        self._init_weights(linear, activation)
        block.append(linear)
        
        # 2. BatchNorm (opcional, solo con suficientes datos)
        if use_batchnorm and activation not in ['sigmoid', 'tanh']:
            batchnorm = nn.BatchNorm1d(num_neurons)
            block.append(batchnorm)
        
        # 3. Activación
        act_layer = self._get_activation(activation)
        block.append(act_layer)
        
        # 4. Dropout (después de activación)
        if dropout_rate > 0:
            block.append(nn.Dropout(p=dropout_rate))
        
        # Guardar capa
        layer_index = len(self.layers_list)
        self.layers_list.append(nn.Sequential(*block))
        
        # Info
        bn_str = "+BN" if use_batchnorm and activation not in ['sigmoid', 'tanh'] else ""
        print(f"   ✅ Capa {layer_index}: {inputs_size} → {num_neurons} | {activation}{bn_str} | Drop: {dropout_rate}")

    def _get_last_linear(self):
        """Obtiene la última capa Linear de layers_list"""
        if len(self.layers_list) == 0:
            return None
        
        for module in reversed(self.layers_list[-1]):
            if isinstance(module, nn.Linear):
                return module
        return None

    def _init_weights(self, linear, activation):
        """Inicialización de pesos"""
        if activation in ['relu', 'gelu', 'silu', 'leaky_relu']:
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        elif activation in ['sigmoid', 'tanh']:
            nn.init.xavier_normal_(linear.weight)
        else:
            nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)

    def _get_activation(self, activation):
        """Retorna la capa de activación correspondiente"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def forward(self, inputs):
        """Forward pass"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        elif isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float32)

        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    def train_model(self, x, y, learning_rate=0.01, epochs=5000,
                    patience=500, lr_factor=0.5, lr_patience=200,
                    min_lr=1e-6, batch_size=4, val_split=0.0,
                    clip_norm=1.0, weight_decay=0.0, warmup_epochs=0):
        """Entrena la red con validaciones y restaura los mejores pesos."""
        import copy

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        if x_tensor.ndim != 2:
            raise ValueError(f"x debe ser una matriz 2D; se recibió {tuple(x_tensor.shape)}")
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)
        if len(x_tensor) != len(y_tensor):
            raise ValueError("x e y deben contener el mismo número de ejemplos")
        if len(x_tensor) < 4:
            raise ValueError("Se requieren al menos 4 ejemplos para entrenar")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs y learning_rate deben ser mayores que cero")

        batch_size = max(2, min(int(batch_size), len(x_tensor)))
        n = len(x_tensor)
        n_val = int(round(n * val_split)) if val_split > 0 else 0
        if n_val >= n:
            n_val = n - 2
        if val_split > 0:
            n_val = max(2, n_val)

        indices = torch.randperm(n)
        if n_val > 0:
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
            x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]
        else:
            x_train, y_train = x_tensor, y_tensor
            x_val, y_val = x_tensor, y_tensor

        if len(x_train) < 2:
            raise ValueError("El conjunto de entrenamiento quedó demasiado pequeño")

        print(f"📊 Train: {len(x_train)} | Val: {len(x_val)} | Batch: {batch_size}")

        if weight_decay > 0:
            optimizer = optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        criterion = nn.MSELoss()

        mejor_loss = float('inf')
        mejor_estado = copy.deepcopy(self.state_dict())
        epocas_sin_mejora = 0
        self.loss_list = []
        self.val_loss_list = []

        for epoch in range(int(epochs)):
            self.train()
            perm = torch.randperm(len(x_train))
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            total_loss = 0.0
            n_batches = 0
            for inicio in range(0, len(x_train), batch_size):
                x_batch = x_shuffled[inicio:inicio + batch_size]
                y_batch = y_shuffled[inicio:inicio + batch_size]
                if x_batch.size(0) == 1 and len(x_train) > 1:
                    continue

                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = criterion(output, y_batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("La pérdida dejó de ser finita durante el entrenamiento")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

            if n_batches == 0:
                raise RuntimeError("No se pudo formar ningún lote de entrenamiento válido")

            total_loss /= n_batches
            self.loss_list.append(total_loss)

            self.eval()
            with torch.no_grad():
                val_output = self.forward(x_val)
                val_loss = float(criterion(val_output, y_val).item())
            self.val_loss_list.append(val_loss)
            scheduler.step(val_loss)

            if epoch % 100 == 0 or epoch == epochs - 1:
                lr_actual = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d} | Loss: {total_loss:.6f} | "
                    f"Val: {val_loss:.6f} | LR: {lr_actual:.6f}"
                )

            if val_loss < mejor_loss - 1e-7:
                mejor_loss = val_loss
                mejor_estado = copy.deepcopy(self.state_dict())
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1

            if epocas_sin_mejora >= patience:
                print(
                    f"\n⏹️ Early stopping en época {epoch} | "
                    f"Best Val Loss: {mejor_loss:.6f}"
                )
                break

        self.load_state_dict(mejor_estado)
        self.eval()
        print(
            f"✅ Entrenamiento completado | Último Loss: {total_loss:.6f} | "
            f"Best Val: {mejor_loss:.6f}"
        )
        return self

    def predict(self, x):
        """Predicción con manejo seguro de tensores"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, (np.ndarray, list)):
                x = torch.tensor(x, dtype=torch.float32)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
                output = self.forward(x)
                return output.squeeze(0).cpu().numpy()
            
            output = self.forward(x)
        return output.cpu().numpy()

    def save(self, filename="model.json"):
        """Guardar modelo"""
        model_data = {"layers": []}

        for i, layer_block in enumerate(self.layers_list):
            linear = None
            activation = "relu"
            dropout_rate = 0.0
            use_batchnorm = False

            for module in layer_block:
                if isinstance(module, nn.Linear):
                    linear = module
                elif isinstance(module, nn.BatchNorm1d):
                    use_batchnorm = True
                elif isinstance(module, nn.ReLU):
                    activation = "relu"
                elif isinstance(module, nn.GELU):
                    activation = "gelu"
                elif isinstance(module, nn.SiLU):
                    activation = "silu"
                elif isinstance(module, nn.LeakyReLU):
                    activation = "leaky_relu"
                elif isinstance(module, nn.Sigmoid):
                    activation = "sigmoid"
                elif isinstance(module, nn.Tanh):
                    activation = "tanh"
                elif isinstance(module, nn.Dropout):
                    dropout_rate = module.p

            if linear is not None:
                layer_data = {
                    "num_neurons": linear.out_features,
                    "input_size": linear.in_features,
                    "activation": activation,
                    "dropout_rate": dropout_rate,
                    "use_batchnorm": use_batchnorm,
                    "weights": linear.weight.detach().cpu().numpy().tolist(),
                    "bias": linear.bias.detach().cpu().numpy().tolist()
                }
                model_data["layers"].append(layer_data)

        ruta = Path(filename)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_suffix(ruta.suffix + ".tmp")
        with temporal.open("w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=4)
        temporal.replace(ruta)
        print(f"💾 Modelo guardado en {ruta} ({len(model_data['layers'])} capas)")

    def load(self, filename="model.json"):
        """Cargar modelo"""
        with open(filename, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.layers_list = nn.ModuleList()

        for layer_data in model_data["layers"]:
            num_neurons = layer_data["num_neurons"]
            inputs_size = layer_data["input_size"]
            activation = layer_data.get("activation", "relu")
            dropout_rate = layer_data.get("dropout_rate", 0.0)
            use_batchnorm = layer_data.get("use_batchnorm", False)

            self.add_layer(num_neurons, inputs_size, activation, dropout_rate, use_batchnorm)

            # Cargar pesos
            linear = None
            for module in self.layers_list[-1]:
                if isinstance(module, nn.Linear):
                    linear = module
                    break

            if linear is not None:
                linear.weight = nn.Parameter(
                    torch.tensor(layer_data["weights"], dtype=torch.float32)
                )
                linear.bias = nn.Parameter(
                    torch.tensor(layer_data["bias"], dtype=torch.float32)
                )

        print(f"📂 Modelo cargado desde {filename} ({len(model_data['layers'])} capas)")


# ============================================================
# PRUEBA
# ============================================================
if __name__ == "__main__":
    print("="*50)
    print("   PRUEBA DE NEURALNETWORK - XOR")
    print("="*50)
    
    # Arquitectura simple para XOR (sin BatchNorm, sin weight decay)
    nn_model = NeuralNetwork()
    nn_model.add_layer(num_neurons=8, inputs_size=2, activation='relu', dropout_rate=0.0)
    nn_model.add_layer(num_neurons=1, activation='sigmoid')

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("\n--- Entrenando XOR ---")
    nn_model.train_model(
        X_train, Y_train, 
        learning_rate=0.01, 
        epochs=10000,
        patience=2000, 
        lr_factor=0.5, 
        lr_patience=500,
        batch_size=2, 
        val_split=0.0, 
        clip_norm=1.0
    )
    
    print("\n--- Predicciones ---")
    predicciones = nn_model.predict(X_train)
    for i in range(len(X_train)):
        print(f"   {X_train[i]} → Esperado: {Y_train[i][0]} → Predicho: {predicciones[i][0]:.4f}")