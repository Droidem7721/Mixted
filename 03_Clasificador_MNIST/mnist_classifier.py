import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CapaConvolucional:
    """Capa convolucional simple (implementación básica)."""
    
    def __init__(self, n_filtros, tam_kernel):
        self.n_filtros = n_filtros
        self.tam_kernel = tam_kernel
        self.filtros = np.random.randn(n_filtros, tam_kernel, tam_kernel) * 0.01
    
    def forward(self, X):
        """Forward pass simplificado."""
        # Flatten para demostración
        return X.reshape(X.shape[0], -1)


class ClasificadorMNIST:
    """Clasificador simple para dígitos MNIST."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.pesos = None
        self.sesgos = None
    
    def sigmoid(self, z):
        """Función sigmoide."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def entrenar(self, X, y, epochs=100):
        """Entrenar el clasificador con regresión logística."""
        n_samples, n_features = X.shape
        n_clases = len(np.unique(y))
        
        # Inicializar pesos
        self.pesos = np.random.randn(n_features, n_clases) * 0.01
        self.sesgos = np.zeros((1, n_clases))
        
        # One-hot encoding
        Y = np.zeros((n_samples, n_clases))
        for i in range(n_samples):
            Y[i, y[i]] = 1
        
        for epoch in range(epochs):
            # Forward
            z = np.dot(X, self.pesos) + self.sesgos
            a = self.sigmoid(z)
            
            # Backward
            error = a - Y
            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error, axis=0, keepdims=True) / n_samples
            
            # Actualizar
            self.pesos -= self.learning_rate * dw
            self.sesgos -= self.learning_rate * db
            
            if epoch % 20 == 0:
                loss = -np.mean(Y * np.log(a + 1e-8) + (1-Y) * np.log(1-a + 1e-8))
                print(f"Época {epoch}, Loss: {loss:.4f}")
    
    def predecir(self, X):
        """Hacer predicciones."""
        z = np.dot(X, self.pesos) + self.sesgos
        a = self.sigmoid(z)
        return np.argmax(a, axis=1)
    
    def evaluar(self, X, y):
        """Calcular precisión."""
        predicciones = self.predecir(X)
        precision = np.mean(predicciones == y)
        return precision


if __name__ == "__main__":
    # Cargar dataset MNIST (versión reducida de sklearn)
    print("Cargando dataset...")
    datos = load_digits()
    X, y = datos.data, datos.target
    
    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar clasificador
    print("Entrenando clasificador...")
    clasificador = ClasificadorMNIST(learning_rate=0.1)
    clasificador.entrenar(X_train, y_train, epochs=100)
    
    # Evaluar
    precision_train = clasificador.evaluar(X_train, y_train)
    precision_test = clasificador.evaluar(X_test, y_test)
    
    print(f"\nPrecisión en entrenamiento: {precision_train:.4f}")
    print(f"Precisión en prueba: {precision_test:.4f}")
