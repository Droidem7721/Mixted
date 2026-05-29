import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    """Autoencoder para aprender representaciones latentes."""
    
    def __init__(self, dim_entrada, dim_latente):
        self.dim_entrada = dim_entrada
        self.dim_latente = dim_latente
        
        # Encoder
        self.W_encoder = np.random.randn(dim_entrada, dim_latente) * 0.01
        self.b_encoder = np.zeros((1, dim_latente))
        
        # Decoder
        self.W_decoder = np.random.randn(dim_latente, dim_entrada) * 0.01
        self.b_decoder = np.zeros((1, dim_entrada))
    
    def relu(self, x):
        """Función ReLU."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivada de ReLU."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Función sigmoide."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, X):
        """Codificar a espacio latente."""
        z = np.dot(X, self.W_encoder) + self.b_encoder
        self.h = self.relu(z)
        self.z = z
        return self.h
    
    def decode(self, h):
        """Decodificar desde espacio latente."""
        z_out = np.dot(h, self.W_decoder) + self.b_decoder
        X_recon = self.sigmoid(z_out)
        return X_recon
    
    def forward(self, X):
        """Forward pass completo."""
        h = self.encode(X)
        X_recon = self.decode(h)
        return X_recon
    
    def entrenar(self, X, epochs=1000, learning_rate=0.01):
        """Entrenar el autoencoder."""
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Forward
            h = self.encode(X)
            X_recon = self.decode(h)
            
            # Backward
            dX_recon = (X_recon - X) * X_recon * (1 - X_recon)
            dW_decoder = np.dot(h.T, dX_recon) / m
            db_decoder = np.sum(dX_recon, axis=0, keepdims=True) / m
            
            dh = np.dot(dX_recon, self.W_decoder.T) * self.relu_derivative(self.z)
            dW_encoder = np.dot(X.T, dh) / m
            db_encoder = np.sum(dh, axis=0, keepdims=True) / m
            
            # Actualizar
            self.W_encoder -= learning_rate * dW_encoder
            self.b_encoder -= learning_rate * db_encoder
            self.W_decoder -= learning_rate * dW_decoder
            self.b_decoder -= learning_rate * db_decoder
            
            if epoch % 100 == 0:
                loss = np.mean((X_recon - X) ** 2)
                print(f"Época {epoch}, MSE: {loss:.6f}")
    
    def obtener_latente(self, X):
        """Obtener representación latente."""
        return self.encode(X)
    
    def reconstruir(self, X):
        """Reconstruir entrada."""
        return self.forward(X)


if __name__ == "__main__":
    # Crear datos de ejemplo (patrones simples)
    print("Generando datos...")
    X = np.random.binomial(1, 0.3, (100, 20)).astype(float)
    
    # Crear y entrenar autoencoder
    print("Entrenando autoencoder...")
    ae = Autoencoder(dim_entrada=20, dim_latente=5)
    ae.entrenar(X, epochs=500, learning_rate=0.1)
    
    # Obtener representaciones latentes
    representaciones_latentes = ae.obtener_latente(X[:5])
    print("\nRepresentaciones latentes (primeras 5 muestras):")
    print(representaciones_latentes)
    
    # Reconstruir
    X_recon = ae.reconstruir(X[:5])
    print("\nError de reconstrucción:")
    error = np.mean((X_recon - X[:5]) ** 2)
    print(f"MSE: {error:.6f}")
