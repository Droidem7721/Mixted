import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class CapaDensaAvanzada:
    """Capa densa con técnicas avanzadas de regularización."""
    
    def __init__(self, n_entrada, n_neuronas, activacion='relu', dropout=0.0):
        self.n_entrada = n_entrada
        self.n_neuronas = n_neuronas
        self.activacion = activacion
        self.dropout = dropout
        
        # Inicialización He para ReLU
        self.pesos = np.random.randn(n_entrada, n_neuronas) * np.sqrt(2.0 / n_entrada)
        self.sesgos = np.zeros((1, n_neuronas))
        
        # Momentum para optimización
        self.v_pesos = np.zeros_like(self.pesos)
        self.v_sesgos = np.zeros_like(self.sesgos)
        self.momentum = 0.9
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X, training=True):
        """Forward con dropout."""
        self.X = X
        self.z = np.dot(X, self.pesos) + self.sesgos
        
        if self.activacion == 'relu':
            self.a = self.relu(self.z)
        elif self.activacion == 'sigmoid':
            self.a = self.sigmoid(self.z)
        elif self.activacion == 'tanh':
            self.a = self.tanh(self.z)
        elif self.activacion == 'linear':
            self.a = self.z
        
        # Aplicar dropout
        if training and self.dropout > 0:
            self.mascara_dropout = np.random.binomial(1, 1 - self.dropout, self.a.shape) / (1 - self.dropout)
            self.a = self.a * self.mascara_dropout
        else:
            self.mascara_dropout = np.ones_like(self.a)
        
        return self.a
    
    def backward(self, dz, learning_rate, l2_lambda=0.0):
        """Backward con momentum y regularización L2."""
        m = self.X.shape[0]
        
        dz = dz * self.mascara_dropout
        
        # Gradientes
        dw = np.dot(self.X.T, dz) / m + (l2_lambda / m) * self.pesos
        db = np.sum(dz, axis=0, keepdims=True) / m
        dX = np.dot(dz, self.pesos.T)
        
        # Momentum
        self.v_pesos = self.momentum * self.v_pesos - learning_rate * dw
        self.v_sesgos = self.momentum * self.v_sesgos - learning_rate * db
        
        self.pesos += self.v_pesos
        self.sesgos += self.v_sesgos
        
        return dX


class RedNeuronalAvanzada:
    """Red neuronal con técnicas avanzadas."""
    
    def __init__(self, capas):
        self.capas = capas
        self.historial_entrenamiento = {
            'loss': [],
            'val_loss': []
        }
    
    def forward(self, X, training=True):
        """Forward propagation."""
        for capa in self.capas:
            X = capa.forward(X, training=training)
        return X
    
    def backward(self, dz, learning_rate, l2_lambda=0.0):
        """Backward propagation."""
        for capa in reversed(self.capas):
            dz = capa.backward(dz, learning_rate, l2_lambda)
    
    def entropia_cruzada(self, y_true, y_pred):
        """Pérdida de entropía cruzada."""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def entrenar(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, l2_lambda=0.0):
        """Entrenar con validación."""
        for epoch in range(epochs):
            # Forward
            predicciones = self.forward(X_train, training=True)
            
            # Loss
            loss = self.entropia_cruzada(y_train, predicciones)
            
            # Backward
            error = (predicciones - y_train) / X_train.shape[0]
            self.backward(error, learning_rate, l2_lambda)
            
            # Validación
            predicciones_val = self.forward(X_val, training=False)
            val_loss = self.entropia_cruzada(y_val, predicciones_val)
            
            self.historial_entrenamiento['loss'].append(loss)
            self.historial_entrenamiento['val_loss'].append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Época {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def predecir(self, X):
        """Hacer predicciones."""
        return self.forward(X, training=False)
    
    def evaluar_precision(self, X, y):
        """Calcular precisión (para clasificación binaria con threshold 0.5)."""
        predicciones = self.predecir(X)
        predicciones_binarias = (predicciones > 0.5).astype(int)
        precision = np.mean(predicciones_binarias == y)
        return precision


if __name__ == "__main__":
    print("=" * 60)
    print("Red Neuronal Avanzada - Dataset Moons")
    print("=" * 60)
    
    # Generar dataset no linealmente separable
    X, y = make_moons(n_samples=300, noise=0.1)
    y = y.reshape(-1, 1)
    
    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Crear red con dropout
    capa1 = CapaDensaAvanzada(2, 16, activacion='relu', dropout=0.2)
    capa2 = CapaDensaAvanzada(16, 8, activacion='relu', dropout=0.1)
    capa3 = CapaDensaAvanzada(8, 1, activacion='sigmoid', dropout=0.0)
    
    red = RedNeuronalAvanzada([capa1, capa2, capa3])
    
    # Entrenar
    print("\nEntrenando...")
    red.entrenar(X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.1, l2_lambda=0.0001)
    
    # Evaluar
    precision_train = red.evaluar_precision(X_train, y_train)
    precision_val = red.evaluar_precision(X_val, y_val)
    precision_test = red.evaluar_precision(X_test, y_test)
    
    print(f"\nPrecisión en entrenamiento: {precision_train:.4f}")
    print(f"Precisión en validación: {precision_val:.4f}")
    print(f"Precisión en prueba: {precision_test:.4f}")
    
    # Graficar historial
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(red.historial_entrenamiento['loss'], label='Training Loss')
        plt.plot(red.historial_entrenamiento['val_loss'], label='Validation Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Historial de Pérdida')
        
        plt.subplot(1, 2, 2)
        plt.scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1], 
                   label='Clase 0', alpha=0.6)
        plt.scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1], 
                   label='Clase 1', alpha=0.6)
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.legend()
        plt.title('Dataset Moons')
        plt.tight_layout()
        plt.savefig('05_red_avanzada_resultados.png')
        print("\nGráfico guardado como '05_red_avanzada_resultados.png'")
    except Exception as e:
        print(f"No se pudo guardar el gráfico: {e}")
