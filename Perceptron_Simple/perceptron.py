import numpy as np

class PerceptronSimple:
    """Implementación simple de un Perceptrón para clasificación binaria."""
    
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def activation(self, x):
        """Función de activación (paso unitario)."""
        return 1 if x >= 0 else 0
    
    def fit(self, X, y):
        """Entrenar el perceptrón."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Actualizar pesos si hay error
                error = y[idx] - y_predicted
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
    
    def predict(self, X):
        """Hacer predicciones."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])


if __name__ == "__main__":
    # Crear datos simples (AND gate)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])
    
    # Entrenar el perceptrón
    perceptron = PerceptronSimple(learning_rate=0.1, epochs=100)
    perceptron.fit(X, y)
    
    # Hacer predicciones
    predictions = perceptron.predict(X)
    print("Predicciones para AND gate:", predictions)
    print("Valores esperados:", y)
