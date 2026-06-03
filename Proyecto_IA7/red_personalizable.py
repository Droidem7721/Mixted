import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class CapaDensa:
    """Capa densa reutilizable con configuración flexible."""
    
    def __init__(self, n_entrada: int, n_neuronas: int, activacion: str = 'relu', 
                 dropout: float = 0.0, nombre: str = None):
        self.n_entrada = n_entrada
        self.n_neuronas = n_neuronas
        self.activacion = activacion
        self.dropout = dropout
        self.nombre = nombre or f"Capa_{n_entrada}_{n_neuronas}"
        
        # Inicialización inteligente
        escala = np.sqrt(2.0 / n_entrada) if activacion == 'relu' else np.sqrt(1.0 / n_entrada)
        self.pesos = np.random.randn(n_entrada, n_neuronas) * escala
        self.sesgos = np.zeros((1, n_neuronas))
        
        # Para optimización
        self.v_pesos = np.zeros_like(self.pesos)
        self.v_sesgos = np.zeros_like(self.sesgos)
        self.momentum = 0.9
    
    def _activar(self, x: np.ndarray) -> np.ndarray:
        """Aplicar función de activación."""
        if self.activacion == 'relu':
            return np.maximum(0, x)
        elif self.activacion == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activacion == 'tanh':
            return np.tanh(x)
        elif self.activacion == 'linear':
            return x
        else:
            raise ValueError(f"Activación desconocida: {self.activacion}")
    
    def _derivada_activacion(self, x_activado: np.ndarray) -> np.ndarray:
        """Calcular derivada de activación."""
        if self.activacion == 'relu':
            return (x_activado > 0).astype(float)
        elif self.activacion == 'sigmoid':
            return x_activado * (1 - x_activado)
        elif self.activacion == 'tanh':
            return 1 - x_activado ** 2
        else:  # linear
            return np.ones_like(x_activado)
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass con dropout."""
        self.X = X
        self.z = np.dot(X, self.pesos) + self.sesgos
        self.a = self._activar(self.z)
        
        if training and self.dropout > 0:
            self.mascara_dropout = np.random.binomial(1, 1 - self.dropout, self.a.shape) / (1 - self.dropout)
            self.a = self.a * self.mascara_dropout
        else:
            self.mascara_dropout = np.ones_like(self.a)
        
        return self.a
    
    def backward(self, dz: np.ndarray, learning_rate: float, l2_lambda: float = 0.0) -> np.ndarray:
        """Backward pass con momentum y regularización."""
        m = self.X.shape[0]
        dz = dz * self.mascara_dropout
        
        dw = np.dot(self.X.T, dz) / m + (l2_lambda / m) * self.pesos
        db = np.sum(dz, axis=0, keepdims=True) / m
        dX = np.dot(dz, self.pesos.T)
        
        # Momentum
        self.v_pesos = self.momentum * self.v_pesos - learning_rate * dw
        self.v_sesgos = self.momentum * self.v_sesgos - learning_rate * db
        
        self.pesos += self.v_pesos
        self.sesgos += self.v_sesgos
        
        return dX
    
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración de la capa."""
        return {
            'n_entrada': self.n_entrada,
            'n_neuronas': self.n_neuronas,
            'activacion': self.activacion,
            'dropout': self.dropout,
            'nombre': self.nombre
        }
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Obtener parámetros entrenables."""
        return {
            'pesos': self.pesos.copy(),
            'sesgos': self.sesgos.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Cargar parámetros entrenables."""
        self.pesos = params['pesos'].copy()
        self.sesgos = params['sesgos'].copy()


class RedNeuronalPersonalizable:
    """Red neuronal flexible y reutilizable para cualquier propósito."""
    
    def __init__(self, configuracion: List[Dict[str, Any]] = None):
        """
        Inicializar red con configuración personalizada.
        
        Args:
            configuracion: Lista de dicts con configuración de capas
                Ejemplo:
                [
                    {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
                    {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
                    {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid'}
                ]
        """
        self.capas = []
        self.configuracion = configuracion or []
        self.historial = {'entrenamiento_loss': [], 'validacion_loss': []}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nombre_modelo = f"RedNeuronal_{self.timestamp}"
    
    def construir(self, dim_entrada: int) -> None:
        """
        Construir arquitectura de la red basada en configuración.
        
        Args:
            dim_entrada: Número de características de entrada
        """
        if not self.configuracion:
            raise ValueError("Configuración vacía. Establece la configuración antes de construir.")
        
        entrada_actual = dim_entrada
        
        for i, config in enumerate(self.configuracion):
            if config['tipo'] != 'densa':
                raise NotImplementedError(f"Tipo de capa no soportado: {config['tipo']}")
            
            n_neuronas = config['neuronas']
            activacion = config.get('activacion', 'relu')
            dropout = config.get('dropout', 0.0)
            nombre = config.get('nombre', f"capa_{i}")
            
            capa = CapaDensa(entrada_actual, n_neuronas, activacion, dropout, nombre)
            self.capas.append(capa)
            entrada_actual = n_neuronas
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward propagation."""
        for capa in self.capas:
            X = capa.forward(X, training=training)
        return X
    
    def backward(self, dz: np.ndarray, learning_rate: float, l2_lambda: float = 0.0) -> None:
        """Backward propagation."""
        for capa in reversed(self.capas):
            dz = capa.backward(dz, learning_rate, l2_lambda)
    
    def _perdida_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Error cuadrático medio."""
        return np.mean((y_pred - y_true) ** 2)
    
    def _perdida_entropia_cruzada(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Entropía cruzada binaria."""
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                epochs: int = 100, learning_rate: float = 0.01, 
                batch_size: int = 32, l2_lambda: float = 0.0,
                tipo_perdida: str = 'mse', verbose: int = 10) -> Dict[str, List[float]]:
        """
        Entrenar la red neuronal.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación (opcional)
            epochs: Número de épocas
            learning_rate: Tasa de aprendizaje
            batch_size: Tamaño del lote
            l2_lambda: Coeficiente de regularización L2
            tipo_perdida: 'mse' o 'entropia_cruzada'
            verbose: Mostrar progreso cada N épocas
        """
        n_muestras = X_train.shape[0]
        fn_perdida = self._perdida_mse if tipo_perdida == 'mse' else self._perdida_entropia_cruzada
        
        if X_val is None:
            X_val, y_val = X_train, y_train
        
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento - Modelo: {self.nombre_modelo}")
        print(f"Configuración: {len(self.capas)} capas, LR: {learning_rate}, Batch: {batch_size}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Mezclar datos
            indices = np.random.permutation(n_muestras)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_muestras, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                predicciones = self.forward(X_batch, training=True)
                error = (predicciones - y_batch) / len(y_batch)
                self.backward(error, learning_rate, l2_lambda)
            
            # Evaluación
            pred_train = self.forward(X_train, training=False)
            loss_train = fn_perdida(y_train, pred_train)
            
            pred_val = self.forward(X_val, training=False)
            loss_val = fn_perdida(y_val, pred_val)
            
            self.historial['entrenamiento_loss'].append(loss_train)
            self.historial['validacion_loss'].append(loss_val)
            
            if (epoch + 1) % verbose == 0:
                print(f"Época {epoch + 1}/{epochs} | Loss: {loss_train:.6f} | Val Loss: {loss_val:.6f}")
        
        print(f"\n✓ Entrenamiento completado\n")
        return self.historial
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Hacer predicciones."""
        return self.forward(X, training=False)
    
    def evaluar(self, X: np.ndarray, y: np.ndarray, tipo_perdida: str = 'mse') -> Dict[str, float]:
        """Evaluar modelo."""
        fn_perdida = self._perdida_mse if tipo_perdida == 'mse' else self._perdida_entropia_cruzada
        predicciones = self.predecir(X)
        loss = fn_perdida(y, predicciones)
        
        return {
            'loss': loss,
            'tipo_perdida': tipo_perdida
        }
    
    def guardar(self, ruta: str = None) -> str:
        """Guardar modelo entrenado."""
        ruta = ruta or f"modelos/{self.nombre_modelo}.pkl"
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        
        modelo_data = {
            'nombre_modelo': self.nombre_modelo,
            'configuracion': self.configuracion,
            'capas_params': [capa.get_params() for capa in self.capas],
            'capas_config': [capa.get_config() for capa in self.capas],
            'historial': self.historial,
            'timestamp': self.timestamp
        }
        
        with open(ruta, 'wb') as f:
            pickle.dump(modelo_data, f)
        
        print(f"✓ Modelo guardado en: {ruta}")
        return ruta
    
    @staticmethod
    def cargar(ruta: str) -> 'RedNeuronalPersonalizable':
        """Cargar modelo guardado."""
        with open(ruta, 'rb') as f:
            modelo_data = pickle.load(f)
        
        red = RedNeuronalPersonalizable(modelo_data['configuracion'])
        red.nombre_modelo = modelo_data['nombre_modelo']
        red.timestamp = modelo_data['timestamp']
        red.historial = modelo_data['historial']
        
        # Reconstruir capas con parámetros
        red.capas = []
        for config in modelo_data['capas_config']:
            capa = CapaDensa(
                config['n_entrada'],
                config['n_neuronas'],
                config['activacion'],
                config['dropout'],
                config['nombre']
            )
            red.capas.append(capa)
        
        # Cargar parámetros
        for capa, params in zip(red.capas, modelo_data['capas_params']):
            capa.set_params(params)
        
        print(f"✓ Modelo cargado desde: {ruta}")
        return red
    
    def obtener_resumen(self) -> str:
        """Obtener resumen de la arquitectura."""
        resumen = f"\n{'='*60}\n"
        resumen += f"Modelo: {self.nombre_modelo}\n"
        resumen += f"Capas: {len(self.capas)}\n"
        resumen += f"{'-'*60}\n"
        
        for i, capa in enumerate(self.capas):
            config = capa.get_config()
            resumen += f"Capa {i+1} ({config['nombre']}): "
            resumen += f"{config['n_entrada']} → {config['n_neuronas']} | "
            resumen += f"Activación: {config['activacion']} | "
            resumen += f"Dropout: {config['dropout']}\n"
        
        resumen += f"{'='*60}\n"
        return resumen


if __name__ == "__main__":
    # Ejemplo 1: Generador de datos de prueba
    print("Generando datos de ejemplo...")
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (np.sum(X[:, :2], axis=1) > 0).astype(float).reshape(-1, 1)
    
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Ejemplo 2: Definir arquitectura personalizada
    config = [
        {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2, 'nombre': 'entrada'},
        {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1, 'nombre': 'oculta'},
        {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid', 'nombre': 'salida'}
    ]
    
    # Ejemplo 3: Crear y entrenar red
    red = RedNeuronalPersonalizable(config)
    red.construir(dim_entrada=4)
    
    print(red.obtener_resumen())
    
    red.entrenar(
        X_train, y_train, 
        X_val, y_val,
        epochs=50, 
        learning_rate=0.1, 
        batch_size=16,
        l2_lambda=0.0001,
        tipo_perdida='entropia_cruzada',
        verbose=10
    )
    
    # Ejemplo 4: Evaluar
    print("\nEvaluando modelo...")
    resultado = red.evaluar(X_test, y_test, tipo_perdida='entropia_cruzada')
    print(f"Pérdida en prueba: {resultado['loss']:.6f}")
    
    # Ejemplo 5: Guardar y cargar
    ruta_guardado = red.guardar()
    red_cargada = RedNeuronalPersonalizable.cargar(ruta_guardado)
    print(red_cargada.obtener_resumen())
