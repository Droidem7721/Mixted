import numpy as np

class CapaDensa:
    """Capa densa (fully connected) con funciones de activación."""
    
    def __init__(self, n_entrada, n_neuronas, activacion='relu'):
        self.n_entrada = n_entrada
        self.n_neuronas = n_neuronas
        self.activacion = activacion
        
        # Inicializar pesos y sesgos
        self.pesos = np.random.randn(n_entrada, n_neuronas) * 0.01
        self.sesgos = np.zeros((1, n_neuronas))
        
        # Para backpropagation
        self.cache = None
    
    def forward(self, X):
        """Forward propagation."""
        self.X = X
        self.z = np.dot(X, self.pesos) + self.sesgos
        
        if self.activacion == 'relu':
            self.a = np.maximum(0, self.z)
        elif self.activacion == 'sigmoid':
            self.a = 1 / (1 + np.exp(-self.z))
        elif self.activacion == 'linear':
            self.a = self.z
        
        return self.a
    
    def backward(self, dz, learning_rate):
        """Backward propagation."""
        m = self.X.shape[0]
        
        # Gradientes
        dw = np.dot(self.X.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        dX = np.dot(dz, self.pesos.T)
        
        # Actualizar pesos y sesgos
        self.pesos -= learning_rate * dw
        self.sesgos -= learning_rate * db
        
        return dX


class RedNeuronalMulticapa:
    """Red neuronal multicapa simple."""
    
    def __init__(self, capas):
        self.capas = capas
    
    def forward(self, X):
        """Forward propagation a través de todas las capas."""
        for capa in self.capas:
            X = capa.forward(X)
        return X
    
    def backward(self, dz, learning_rate):
        """Backward propagation a través de todas las capas."""
        for capa in reversed(self.capas):
            dz = capa.backward(dz, learning_rate)
    
    def entrenar(self, X, y, epochs=1000, learning_rate=0.01):
        """Entrenar la red."""
        for epoch in range(epochs):
            # Forward
            predicciones = self.forward(X)
            
            # Error
            error = predicciones - y
            
            # Backward
            self.backward(error, learning_rate)
            
            if epoch % 100 == 0:
                mse = np.mean(error ** 2)
                print(f"Época {epoch}, MSE: {mse:.4f}")
    
    def predecir(self, X):
        """Hacer predicciones."""
        return self.forward(X)


if __name__ == "__main__":
    # Datos de ejemplo (XOR problem)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    # Crear red neuronal
    capa1 = CapaDensa(2, 4, activacion='relu')
    capa2 = CapaDensa(4, 1, activacion='sigmoid')
    
    red = RedNeuronalMulticapa([capa1, capa2])
    
    # Entrenar
    red.entrenar(X, y, epochs=1000, learning_rate=0.1)
    
    # Predicciones
    predicciones = red.predecir(X)
    print("\nPredicciones para XOR:")
    for i in range(len(X)):
        print(f"{X[i]} -> {predicciones[i][0]:.4f}")
