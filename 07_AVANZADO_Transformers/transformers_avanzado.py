"""Transformers y Redes Neuronales Avanzadas - Nivel Expert."""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.special import softmax as scipy_softmax
import pickle
from pathlib import Path


class MecanismoAtencion:
    """Mecanismo de Atención Multi-Cabeza (Multi-Head Attention)."""
    
    def __init__(self, dim_modelo: int, n_cabezas: int):
        assert dim_modelo % n_cabezas == 0, "dim_modelo debe ser divisible por n_cabezas"
        
        self.dim_modelo = dim_modelo
        self.n_cabezas = n_cabezas
        self.dim_cabeza = dim_modelo // n_cabezas
        
        # Pesos para Query, Key, Value
        self.W_Q = np.random.randn(dim_modelo, dim_modelo) * 0.01
        self.W_K = np.random.randn(dim_modelo, dim_modelo) * 0.01
        self.W_V = np.random.randn(dim_modelo, dim_modelo) * 0.01
        self.W_out = np.random.randn(dim_modelo, dim_modelo) * 0.01
    
    def atencion_escalada_puntual(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                   mascara: np.ndarray = None) -> np.ndarray:
        """Scaled Dot-Product Attention."""
        puntajes = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.dim_cabeza)
        
        if mascara is not None:
            puntajes = puntajes + (mascara * -1e9)
        
        pesos_atencion = scipy_softmax(puntajes, axis=-1)
        salida = np.matmul(pesos_atencion, V)
        
        return salida, pesos_atencion
    
    def dividir_cabezas(self, X: np.ndarray) -> np.ndarray:
        """Dividir en múltiples cabezas."""
        batch_size, seq_len, _ = X.shape
        X = X.reshape(batch_size, seq_len, self.n_cabezas, self.dim_cabeza)
        return X.transpose(0, 2, 1, 3)
    
    def combinar_cabezas(self, X: np.ndarray) -> np.ndarray:
        """Combinar cabezas."""
        batch_size, _, seq_len, dim_cabeza = X.shape
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(batch_size, seq_len, self.n_cabezas * dim_cabeza)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
               mascara: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass."""
        batch_size = Q.shape[0]
        
        # Proyectar
        Q = np.matmul(Q, self.W_Q)
        K = np.matmul(K, self.W_K)
        V = np.matmul(V, self.W_V)
        
        # Dividir en cabezas
        Q = self.dividir_cabezas(Q)
        K = self.dividir_cabezas(K)
        V = self.dividir_cabezas(V)
        
        # Atención
        salida_atencion, pesos = self.atencion_escalada_puntual(Q, K, V, mascara)
        
        # Combinar cabezas
        salida = self.combinar_cabezas(salida_atencion)
        
        # Proyección de salida
        salida = np.matmul(salida, self.W_out)
        
        return salida, pesos


class CapaRedNeuronal:
    """Capa de Red Neuronal Feed-Forward (FFN)."""
    
    def __init__(self, dim_modelo: int, dim_oculta: int):
        self.W1 = np.random.randn(dim_modelo, dim_oculta) * 0.01
        self.b1 = np.zeros((1, dim_oculta))
        self.W2 = np.random.randn(dim_oculta, dim_modelo) * 0.01
        self.b2 = np.zeros((1, dim_modelo))
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2


class BloqueDeCoder:
    """Bloque de Decoder del Transformer."""
    
    def __init__(self, dim_modelo: int, n_cabezas: int, dim_ffn: int, dropout: float = 0.1):
        self.atencion_propia = MecanismoAtencion(dim_modelo, n_cabezas)
        self.atencion_cruzada = MecanismoAtencion(dim_modelo, n_cabezas)
        self.ffn = CapaRedNeuronal(dim_modelo, dim_ffn)
        
        self.dropout = dropout
        self.eps = 1e-6
        self.gamma1 = np.ones((1, dim_modelo))
        self.beta1 = np.zeros((1, dim_modelo))
        self.gamma2 = np.ones((1, dim_modelo))
        self.beta2 = np.zeros((1, dim_modelo))
        self.gamma3 = np.ones((1, dim_modelo))
        self.beta3 = np.zeros((1, dim_modelo))
    
    def normalizacion_capa(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer Normalization."""
        media = np.mean(X, axis=-1, keepdims=True)
        varianza = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - media) / np.sqrt(varianza + self.eps)
        return gamma * X_norm + beta
    
    def forward(self, X_decoder: np.ndarray, X_encoder: np.ndarray, 
               mascara_propia: np.ndarray = None, 
               mascara_cruzada: np.ndarray = None) -> np.ndarray:
        """Forward pass."""
        # Auto-atención
        atencion1, _ = self.atencion_propia.forward(X_decoder, X_decoder, X_decoder, mascara_propia)
        if self.dropout > 0:
            atencion1 = atencion1 * (np.random.rand(*atencion1.shape) > self.dropout)
        X_decoder = X_decoder + atencion1
        X_decoder = self.normalizacion_capa(X_decoder, self.gamma1, self.beta1)
        
        # Atención cruzada
        atencion2, _ = self.atencion_cruzada.forward(X_decoder, X_encoder, X_encoder, mascara_cruzada)
        if self.dropout > 0:
            atencion2 = atencion2 * (np.random.rand(*atencion2.shape) > self.dropout)
        X_decoder = X_decoder + atencion2
        X_decoder = self.normalizacion_capa(X_decoder, self.gamma2, self.beta2)
        
        # FFN
        ffn_out = self.ffn.forward(X_decoder)
        if self.dropout > 0:
            ffn_out = ffn_out * (np.random.rand(*ffn_out.shape) > self.dropout)
        X_decoder = X_decoder + ffn_out
        X_decoder = self.normalizacion_capa(X_decoder, self.gamma3, self.beta3)
        
        return X_decoder


class TransformerSimplificado:
    """Transformer simplificado para tareas de secuencia-a-secuencia."""
    
    def __init__(self, vocab_size: int, dim_modelo: int = 512, n_capas: int = 2, 
                n_cabezas: int = 8, dim_ffn: int = 2048, max_seq_len: int = 100):
        """
        Inicializar Transformer.
        
        Args:
            vocab_size: Tamaño del vocabulario
            dim_modelo: Dimensión del modelo
            n_capas: Número de capas de decoder
            n_cabezas: Número de cabezas de atención
            dim_ffn: Dimensión de la capa FFN oculta
            max_seq_len: Máxima longitud de secuencia
        """
        self.vocab_size = vocab_size
        self.dim_modelo = dim_modelo
        self.n_capas = n_capas
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embedding_tokens = np.random.randn(vocab_size, dim_modelo) * 0.01
        self.embedding_posiciones = np.random.randn(max_seq_len, dim_modelo) * 0.01
        
        # Capas decoder
        self.capas_decoder = [
            BloqueDeCoder(dim_modelo, n_cabezas, dim_ffn) 
            for _ in range(n_capas)
        ]
        
        # Capa de salida
        self.W_salida = np.random.randn(dim_modelo, vocab_size) * 0.01
        self.b_salida = np.zeros((1, vocab_size))
    
    def generar_mascara_causal(self, seq_len: int) -> np.ndarray:
        """Generar máscara causal para atención."""
        mascara = np.tril(np.ones((seq_len, seq_len)))
        return (1 - mascara)
    
    def forward(self, tokens: np.ndarray, encoder_output: np.ndarray = None) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            tokens: IDs de tokens (batch_size, seq_len)
            encoder_output: Salida del encoder para atención cruzada
        
        Returns:
            Logits de salida (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        
        # Embeddings
        X = self.embedding_tokens[tokens]  # (batch_size, seq_len, dim_modelo)
        X = X + self.embedding_posiciones[:seq_len]
        
        # Normalización
        X = (X - np.mean(X)) / (np.std(X) + 1e-6)
        
        # Máscara causal
        mascara_causal = self.generar_mascara_causal(seq_len)
        
        # Pasar por capas decoder
        if encoder_output is None:
            encoder_output = X
        
        for capa in self.capas_decoder:
            X = capa.forward(X, encoder_output, mascara_causal, None)
        
        # Proyección a vocabulario
        logits = np.dot(X, self.W_salida) + self.b_salida
        
        return logits
    
    def generar(self, tokens_iniciales: np.ndarray, max_tokens: int = 50, 
               temperatura: float = 1.0) -> np.ndarray:
        """
        Generar secuencia autorreégresiva.
        
        Args:
            tokens_iniciales: Tokens iniciales (batch_size, seq_inicial)
            max_tokens: Máximo número de tokens a generar
            temperatura: Control de aleatoriedad (>1 más aleatorio, <1 más determinístico)
        
        Returns:
            Secuencia generada
        """
        secuencia = tokens_iniciales.copy()
        
        for _ in range(max_tokens):
            # Forward
            logits = self.forward(secuencia)
            
            # Último token
            logits_siguiente = logits[:, -1, :] / temperatura
            
            # Softmax
            probs = scipy_softmax(logits_siguiente, axis=-1)
            
            # Muestreo
            siguiente_token = np.array([
                np.random.choice(self.vocab_size, p=probs[i])
                for i in range(probs.shape[0])
            ]).reshape(-1, 1)
            
            secuencia = np.concatenate([secuencia, siguiente_token], axis=1)
            
            if secuencia.shape[1] >= self.max_seq_len:
                break
        
        return secuencia
    
    def guardar(self, ruta: str) -> None:
        """Guardar modelo."""
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Modelo guardado en: {ruta}")
    
    @staticmethod
    def cargar(ruta: str) -> 'TransformerSimplificado':
        """Cargar modelo."""
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        print(f"✓ Modelo cargado desde: {ruta}")
        return modelo


class RedNeuronalGrafos:
    """Red Neuronal para Grafos (Graph Neural Network - GNN)."""
    
    def __init__(self, dim_entrada: int, dim_oculta: int, n_capas: int = 2):
        """
        Inicializar GNN.
        
        Args:
            dim_entrada: Dimensión de features de nodos
            dim_oculta: Dimensión de estados ocultos
            n_capas: Número de capas de convolución de grafos
        """
        self.dim_entrada = dim_entrada
        self.dim_oculta = dim_oculta
        self.n_capas = n_capas
        
        # Matrices de transformación
        self.W = [
            np.random.randn(dim_entrada if i == 0 else dim_oculta, dim_oculta) * 0.01
            for i in range(n_capas)
        ]
        
        # Sesgos
        self.b = [np.zeros((1, dim_oculta)) for _ in range(n_capas)]
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)
    
    def convolucionar_grafo(self, X: np.ndarray, matriz_adyacencia: np.ndarray, 
                           W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convolución de grafo.
        X: features de nodos (n_nodos, dim_features)
        matriz_adyacencia: matriz de adyacencia (n_nodos, n_nodos)
        """
        # Normalizar matriz de adyacencia
        D = np.diag(np.sum(matriz_adyacencia, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-8))
        A_norm = D_inv_sqrt @ matriz_adyacencia @ D_inv_sqrt
        
        # Convolucionar
        X_conv = A_norm @ X @ W + b
        return X_conv
    
    def forward(self, X: np.ndarray, matriz_adyacencia: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Features de nodos (n_nodos, dim_entrada)
            matriz_adyacencia: Matriz de adyacencia (n_nodos, n_nodos)
        
        Returns:
            Embeddings de nodos (n_nodos, dim_oculta)
        """
        for i in range(self.n_capas):
            X = self.convolucionar_grafo(X, matriz_adyacencia, self.W[i], self.b[i])
            if i < self.n_capas - 1:
                X = self.relu(X)
        
        return X
    
    def clasificar_nodos(self, X: np.ndarray, matriz_adyacencia: np.ndarray, 
                        n_clases: int, nodos_entrenamiento: np.ndarray = None) -> np.ndarray:
        """
        Clasificación de nodos.
        
        Args:
            X: Features de nodos
            matriz_adyacencia: Matriz de adyacencia
            n_clases: Número de clases
            nodos_entrenamiento: Índices de nodos para entrenar
        
        Returns:
            Predicciones de clase
        """
        embeddings = self.forward(X, matriz_adyacencia)
        W_clase = np.random.randn(self.dim_oculta, n_clases) * 0.01
        logits = embeddings @ W_clase
        return np.argmax(logits, axis=1)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Transformers y Redes Neuronales Avanzadas")
    print("="*70)
    
    # Ejemplo 1: Transformer Simplificado
    print("\n[1] Inicializando Transformer...")
    vocab_size = 1000
    transformer = TransformerSimplificado(
        vocab_size=vocab_size,
        dim_modelo=256,
        n_capas=2,
        n_cabezas=4,
        dim_ffn=512
    )
    print(f✓ Transformer creado: {vocab_size} tokens, 2 capas, 4 cabezas")
    
    # Ejemplo 2: Forward pass
    print("\n[2] Forward pass...")
    batch_size, seq_len = 4, 10
    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = transformer.forward(tokens)
    print(f✓ Logits: {logits.shape}")
    
    # Ejemplo 3: Generación de texto
    print("\n[3] Generando secuencia...")
    tokens_iniciales = np.array([[1, 2, 3]])
    secuencia_generada = transformer.generar(tokens_iniciales, max_tokens=20, temperatura=1.0)
    print(f✓ Secuencia generada: {secuencia_generada.shape}")
    
    # Ejemplo 4: GNN
    print("\n[4] Red Neuronal de Grafos...")
    n_nodos = 10
    dim_features = 32
    X_nodos = np.random.randn(n_nodos, dim_features)
    
    # Matriz de adyacencia (grafo aleatorio)
    matriz_adyacencia = (np.random.rand(n_nodos, n_nodos) > 0.7).astype(float)
    
    gnn = RedNeuronalGrafos(dim_features, dim_oculta=16, n_capas=2)
    embeddings = gnn.forward(X_nodos, matriz_adyacencia)
    print(f✓ Embeddings de grafo: {embeddings.shape}")
    
    predicciones = gnn.clasificar_nodos(X_nodos, matriz_adyacencia, n_clases=3)
    print(f✓ Clasificación de nodos: {predicciones.shape}")
    
    print("\n" + "="*70)
