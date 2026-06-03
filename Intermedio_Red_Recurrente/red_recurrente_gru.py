import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

class GRUCell:
    """Celda GRU (Gated Recurrent Unit) simplificada."""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Pesos para puerta de reset
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((1, hidden_size))
        
        # Pesos para puerta de actualización
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((1, hidden_size))
        
        # Pesos para candidato
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X, h_prev):
        """Forward pass de GRU.
        X: entrada de forma (batch_size, input_size)
        h_prev: estado oculto anterior (batch_size, hidden_size)
        """
        # Concatenar entrada y estado anterior
        concatenado = np.hstack([X, h_prev])
        
        # Puerta de reset
        r = self.sigmoid(np.dot(concatenado, self.W_r) + self.b_r)
        
        # Puerta de actualización
        z = self.sigmoid(np.dot(concatenado, self.W_z) + self.b_z)
        
        # Candidato
        concatenado_reset = np.hstack([X, r * h_prev])
        h_tilde = self.tanh(np.dot(concatenado_reset, self.W_h) + self.b_h)
        
        # Nuevo estado
        h = (1 - z) * h_prev + z * h_tilde
        
        # Guardar para backward
        self.cache = (X, h_prev, r, z, h_tilde, concatenado, concatenado_reset)
        
        return h


class RedRecurrente:
    """Red Recurrente simple con GRU."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.gru = GRUCell(input_size, hidden_size)
        
        # Capa de salida
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        """Softmax para clasificación multiclase."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X_secuencia, training=True):
        """Forward pass para una secuencia.
        X_secuencia: lista de entradas (timesteps, batch_size, input_size)
        """
        batch_size = X_secuencia[0].shape[0]
        h = np.zeros((batch_size, self.hidden_size))
        
        self.h_states = [h]
        
        for x in X_secuencia:
            h = self.gru.forward(x, h)
            self.h_states.append(h.copy())
        
        # Salida
        z = np.dot(h, self.W_out) + self.b_out
        output = self.softmax(z)
        
        return output
    
    def predecir(self, X):
        """Predicción."""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)


class ClasificadorSecuencial:
    """Clasificador para datos secuenciales."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.red = RedRecurrente(input_size, hidden_size, output_size)
        self.output_size = output_size
    
    def entropia_cruzada(self, y_true, y_pred):
        """Entropía cruzada."""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def entrenar(self, X_train_seq, y_train, epochs=50, learning_rate=0.01):
        """Entrenar clasificador.
        X_train_seq: lista de secuencias
        y_train: etiquetas (one-hot encoded)
        """
        for epoch in range(epochs):
            predicciones = self.red.forward(X_train_seq, training=True)
            loss = self.entropia_cruzada(y_train, predicciones)
            
            if epoch % 10 == 0:
                print(f"Época {epoch}, Loss: {loss:.4f}")
    
    def evaluar(self, X_seq, y_true):
        """Evaluar precisión."""
        predicciones = self.red.predecir(X_seq)
        y_true_labels = np.argmax(y_true, axis=1)
        precision = np.mean(predicciones == y_true_labels)
        return precision


if __name__ == "__main__":
    print("=" * 60)
    print("Red Recurrente con GRU - Clasificación Secuencial")
    print("=" * 60)
    
    # Crear datos simples (secuencias)
    n_muestras = 100
    seq_length = 5
    input_size = 3
    n_clases = 2
    
    # Generar secuencias de ejemplo
    X_secuencias = []
    y_etiquetas = []
    
    for i in range(n_muestras):
        # Secuencia aleatoria
        seq = np.random.randn(seq_length, input_size)
        # Etiqueta basada en suma de características
        suma = np.sum(seq)
        etiqueta = 1 if suma > 0 else 0
        
        X_secuencias.append(seq)
        y_etiquetas.append(etiqueta)
    
    y_etiquetas = np.array(y_etiquetas)
    
    # One-hot encoding
    y_onehot = np.zeros((len(y_etiquetas), n_clases))
    for i in range(len(y_etiquetas)):
        y_onehot[i, y_etiquetas[i]] = 1
    
    # Dividir datos
    split_idx = int(0.8 * len(X_secuencias))
    X_train = X_secuencias[:split_idx]
    X_test = X_secuencias[split_idx:]
    y_train = y_onehot[:split_idx]
    y_test = y_onehot[split_idx:]
    
    # Crear y entrenar
    print("\nEntrenando clasificador...")
    clasificador = ClasificadorSecuencial(input_size, 8, n_clases)
    clasificador.entrenar(X_train, y_train, epochs=50, learning_rate=0.01)
    
    # Evaluar
    precision_train = clasificador.evaluar(X_train, y_train)
    precision_test = clasificador.evaluar(X_test, y_test)
    
    print(f"\nPrecisión en entrenamiento: {precision_train:.4f}")
    print(f"Precisión en prueba: {precision_test:.4f}")
