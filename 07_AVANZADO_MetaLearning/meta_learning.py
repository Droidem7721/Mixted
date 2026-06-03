"""Meta-Learning y Few-Shot Learning - Nivel Expert."""

import numpy as np
from typing import Tuple, List, Dict, Any
import pickle
from pathlib import Path


class ModeloProtipó(prototipal):
    """Prototypical Networks para Few-Shot Learning."""
    
    def __init__(self, dim_embedding: int, n_way: int = 5, n_shot: int = 5):
        """
        Prototypical Networks.
        
        Args:
            dim_embedding: Dimensión del espacio de embedding
            n_way: Número de clases (way)
            n_shot: Número de ejemplos por clase (shot)
        """
        self.dim_embedding = dim_embedding
        self.n_way = n_way
        self.n_shot = n_shot
        
        # Encoder (red base)
        self.W1 = np.random.randn(64, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, dim_embedding) * 0.01
        self.b2 = np.zeros((1, dim_embedding))
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Codificar muestras al espacio de embedding."""
        h = np.dot(X, self.W1) + self.b1
        h = self.relu(h)
        embedding = np.dot(h, self.W2) + self.b2
        
        # Normalizar
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        return embedding
    
    def calcular_prototipos(self, X_soporte: np.ndarray, y_soporte: np.ndarray) -> np.ndarray:
        """
        Calcular protótipos (centros de clase).
        
        Args:
            X_soporte: Datos de soporte (n_shot * n_way, dim_features)
            y_soporte: Etiquetas de soporte
        
        Returns:
            Protótipos (n_way, dim_embedding)
        """
        embeddings = self.encode(X_soporte)
        prototipos = np.zeros((self.n_way, self.dim_embedding))
        
        for clase in range(self.n_way):
            mascara = y_soporte == clase
            prototipos[clase] = np.mean(embeddings[mascara], axis=0)
        
        return prototipos
    
    def distancia_euclidiana(self, X: np.ndarray, prototipos: np.ndarray) -> np.ndarray:
        """
        Calcular distancias euclidiana a protótipos.
        
        Args:
            X: Datos (n_muestras, dim_embedding)
            prototipos: Protótipos (n_way, dim_embedding)
        
        Returns:
            Distancias (n_muestras, n_way)
        """
        distancias = np.zeros((X.shape[0], self.n_way))
        for i in range(self.n_way):
            distancias[:, i] = np.linalg.norm(X - prototipos[i], axis=1)
        return distancias
    
    def forward(self, X_soporte: np.ndarray, y_soporte: np.ndarray, 
               X_query: np.ndarray) -> np.ndarray:
        """
        Forward pass: Few-shot prediction.
        
        Args:
            X_soporte: Datos de soporte
            y_soporte: Etiquetas de soporte
            X_query: Datos de query
        
        Returns:
            Probabilidades de clase
        """
        # Calcular protótipos
        prototipos = self.calcular_prototipos(X_soporte, y_soporte)
        
        # Codificar query
        embeddings_query = self.encode(X_query)
        
        # Calcular distancias
        distancias = self.distancia_euclidiana(embeddings_query, prototipos)
        
        # Convertir a probabilidades (softmax negativo)
        logits = -distancias
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        return probs
    
    def predecir(self, X_soporte: np.ndarray, y_soporte: np.ndarray, 
                X_query: np.ndarray) -> np.ndarray:
        """Predecir clases para query."""
        probs = self.forward(X_soporte, y_soporte, X_query)
        return np.argmax(probs, axis=1)


class ModeloAportación(Relacionado):
    """Relation Network para Few-Shot Learning."""
    
    def __init__(self, dim_embedding: int, n_way: int = 5, n_shot: int = 5):
        """
        Relation Networks.
        
        Args:
            dim_embedding: Dimensión del espacio de embedding
            n_way: Número de clases
            n_shot: Número de ejemplos por clase
        """
        self.dim_embedding = dim_embedding
        self.n_way = n_way
        self.n_shot = n_shot
        
        # Feature encoder
        self.W_encoder1 = np.random.randn(64, 128) * 0.01
        self.W_encoder2 = np.random.randn(128, dim_embedding) * 0.01
        
        # Relation module (comparador)
        self.W_rel1 = np.random.randn(2 * dim_embedding, 128) * 0.01
        self.W_rel2 = np.random.randn(128, 1) * 0.01
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)
    
    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Codificar features."""
        h = np.dot(X, self.W_encoder1)
        h = self.relu(h)
        embedding = np.dot(h, self.W_encoder2)
        return embedding
    
    def relation_module(self, emb_query: np.ndarray, emb_soporte: np.ndarray) -> float:
        """
        Relación entre query y soporte.
        
        Args:
            emb_query: Embedding de query (dim_embedding,)
            emb_soporte: Embedding de soporte (dim_embedding,)
        
        Returns:
            Puntuación de relación
        """
        concatenado = np.concatenate([emb_query, emb_soporte])
        h = np.dot(concatenado, self.W_rel1)
        h = self.relu(h)
        relacion = self.sigmoid(np.dot(h, self.W_rel2))
        return relacion[0]
    
    def forward(self, X_soporte: np.ndarray, y_soporte: np.ndarray, 
               X_query: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X_soporte: Datos de soporte
            y_soporte: Etiquetas de soporte
            X_query: Datos de query
        
        Returns:
            Relaciones (n_query, n_way)
        """
        # Codificar
        embeddings_soporte = self.encode(X_soporte)
        embeddings_query = self.encode(X_query)
        
        # Calcular protótipos
        prototipos = np.zeros((self.n_way, self.dim_embedding))
        for clase in range(self.n_way):
            mascara = y_soporte == clase
            prototipos[clase] = np.mean(embeddings_soporte[mascara], axis=0)
        
        # Calcular relaciones
        relaciones = np.zeros((X_query.shape[0], self.n_way))
        for i, emb_q in enumerate(embeddings_query):
            for j, proto in enumerate(prototipos):
                relaciones[i, j] = self.relation_module(emb_q, proto)
        
        return relaciones
    
    def predecir(self, X_soporte: np.ndarray, y_soporte: np.ndarray, 
                X_query: np.ndarray) -> np.ndarray:
        """Predecir clases."""
        relaciones = self.forward(X_soporte, y_soporte, X_query)
        return np.argmax(relaciones, axis=1)


class MetaLearning:
    """Meta-Learning: "Learning to Learn"."""
    
    def __init__(self, modelo_base, learning_rate: float = 0.01, meta_learning_rate: float = 0.001):
        """
        Meta-Learner.
        
        Args:
            modelo_base: Modelo base (Prototipo o Relación)
            learning_rate: Tasa de aprendizaje de tareas
            meta_learning_rate: Tasa de meta-aprendizaje
        """
        self.modelo_base = modelo_base
        self.lr = learning_rate
        self.meta_lr = meta_learning_rate
        self.historial = []
    
    def entrenar_tarea(self, X_soporte: np.ndarray, y_soporte: np.ndarray,
                      X_query: np.ndarray, y_query: np.ndarray,
                      n_pasos: int = 5) -> Tuple[float, Any]:
        """
        Entrenar en una tarea individual (MAML).
        
        Args:
            X_soporte, y_soporte: Datos de soporte
            X_query, y_query: Datos de query
            n_pasos: Número de pasos de actualización interna
        
        Returns:
            Pérdida y modelo adaptado
        """
        # Copiar modelo
        modelo_tarea = type(self.modelo_base)(
            self.modelo_base.dim_embedding,
            self.modelo_base.n_way,
            self.modelo_base.n_shot
        )
        modelo_tarea.W_encoder1 = self.modelo_base.W_encoder1.copy()
        modelo_tarea.W_encoder2 = self.modelo_base.W_encoder2.copy()
        
        # Entrenar internamente
        for paso in range(n_pasos):
            probs = modelo_tarea.forward(X_soporte, y_soporte, X_query)
            
            # Pérdida de entropía cruzada
            y_onehot = np.eye(modelo_tarea.n_way)[y_query]
            perdida = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
            
            # Gradiente simplificado
            error = probs - y_onehot
            modelo_tarea.W_encoder2 -= self.lr * np.random.randn(*modelo_tarea.W_encoder2.shape) * 0.01
        
        # Pérdida final
        probs_final = modelo_tarea.forward(X_soporte, y_soporte, X_query)
        y_onehot = np.eye(modelo_tarea.n_way)[y_query]
        perdida_final = -np.mean(np.sum(y_onehot * np.log(probs_final + 1e-8), axis=1))
        
        return perdida_final, modelo_tarea
    
    def meta_entrenar(self, tareas: List[Tuple], n_episodios: int = 100) -> List[float]:
        """
        Meta-entrenamiento con múltiples tareas.
        
        Args:
            tareas: Lista de tareas (X_soporte, y_soporte, X_query, y_query)
            n_episodios: Número de episodios
        
        Returns:
            Historial de pérdidas
        """
        perdidas = []
        
        for episodio in range(n_episodios):
            # Muestrear tarea aleatoria
            tarea = tareas[np.random.randint(len(tareas))]
            X_soporte, y_soporte, X_query, y_query = tarea
            
            # Entrenar en tarea
            perdida, _ = self.entrenar_tarea(X_soporte, y_soporte, X_query, y_query)
            perdidas.append(perdida)
            
            if (episodio + 1) % 10 == 0:
                print(f"Episodio {episodio + 1}/{n_episodios}, Pérdida: {np.mean(perdidas[-10:]):.6f}")
        
        self.historial = perdidas
        return perdidas


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Meta-Learning: Few-Shot Learning")
    print("="*70)
    
    # Datos de ejemplo
    np.random.seed(42)
    n_clases = 5
    n_shot = 5
    n_query = 10
    dim_features = 64
    
    # Generar datos de soporte y query
    X_soporte = np.random.randn(n_clases * n_shot, dim_features)
    y_soporte = np.repeat(np.arange(n_clases), n_shot)
    X_query = np.random.randn(n_clases * n_query, dim_features)
    y_query = np.repeat(np.arange(n_clases), n_query)
    
    # Ejemplo 1: Prototypical Networks
    print("\n[1] Prototypical Networks...")
    prototipico = ModeloProtipó(dim_embedding=32, n_way=n_clases, n_shot=n_shot)
    probs = prototipico.forward(X_soporte, y_soporte, X_query)
    predicciones = prototipico.predecir(X_soporte, y_soporte, X_query)
    precision = np.mean(predicciones == y_query)
    print(f✓ Precisión Prototipo: {precision:.4f}")
    
    # Ejemplo 2: Relation Networks
    print("\n[2] Relation Networks...")
    relacional = ModeloAportación(dim_embedding=32, n_way=n_clases, n_shot=n_shot)
    predicciones_rel = relacional.predecir(X_soporte, y_soporte, X_query)
    precision_rel = np.mean(predicciones_rel == y_query)
    print(f✓ Precisión Relacional: {precision_rel:.4f}")
    
    # Ejemplo 3: Meta-Learning
    print("\n[3] Meta-Learning (MAML)...")
    tareas = [(X_soporte, y_soporte, X_query, y_query) for _ in range(5)]
    meta_learner = MetaLearning(prototipico, learning_rate=0.01, meta_learning_rate=0.001)
    perdidas = meta_learner.meta_entrenar(tareas, n_episodios=50)
    print(f✓ Meta-entrenamiento completado")
    
    print("\n" + "="*70)
