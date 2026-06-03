"""Módulo de utilidades para la red neuronal."""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any


class PreprocesadorDatos:
    """Preprocesor de datos reutilizable."""
    
    def __init__(self):
        self.media = None
        self.desv_est = None
        self.min_val = None
        self.max_val = None
    
    def normalizar_minmax(self, X: np.ndarray) -> np.ndarray:
        """Normalizar a rango [0, 1]."""
        if self.min_val is None:
            self.min_val = np.min(X, axis=0)
            self.max_val = np.max(X, axis=0)
        
        return (X - self.min_val) / (self.max_val - self.min_val + 1e-8)
    
    def estandarizar(self, X: np.ndarray) -> np.ndarray:
        """Estandarizar a media 0 y desviación 1."""
        if self.media is None:
            self.media = np.mean(X, axis=0)
            self.desv_est = np.std(X, axis=0)
        
        return (X - self.media) / (self.desv_est + 1e-8)
    
    def invertir_estandarizacion(self, X: np.ndarray) -> np.ndarray:
        """Invertir estandarización."""
        return X * self.desv_est + self.media


class GeneradorConfiguracion:
    """Generador de configuraciones predefinidas."""
    
    @staticmethod
    def simple() -> List[Dict[str, Any]]:
        """Configuración simple: 1 capa oculta."""
        return [
            {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid'}
        ]
    
    @staticmethod
    def moderada() -> List[Dict[str, Any]]:
        """Configuración moderada: 2 capas ocultas."""
        return [
            {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
            {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid'}
        ]
    
    @staticmethod
    def profunda() -> List[Dict[str, Any]]:
        """Configuración profunda: 3 capas ocultas."""
        return [
            {'tipo': 'densa', 'neuronas': 32, 'activacion': 'relu', 'dropout': 0.3},
            {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
            {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid'}
        ]
    
    @staticmethod
    def multiclase(n_clases: int) -> List[Dict[str, Any]]:
        """Configuración para clasificación multiclase."""
        return [
            {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
            {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': n_clases, 'activacion': 'softmax'}
        ]
    
    @staticmethod
    def regresion() -> List[Dict[str, Any]]:
        """Configuración para regresión."""
        return [
            {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
            {'tipo': 'densa', 'neuronas': 1, 'activacion': 'linear'}
        ]


class EvaluadorMetricas:
    """Calculador de métricas de evaluación."""
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Calcular precisión para clasificación binaria."""
        y_pred_bin = (y_pred > threshold).astype(int)
        return np.mean(y_pred_bin == y_true)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Error absoluto medio."""
        return np.mean(np.abs(y_pred - y_true))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Raíz del error cuadrático medio."""
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
        """Calcular matriz de confusión."""
        y_pred_bin = (y_pred > threshold).astype(int)
        tp = np.sum((y_pred_bin == 1) & (y_true == 1))
        tn = np.sum((y_pred_bin == 0) & (y_true == 0))
        fp = np.sum((y_pred_bin == 1) & (y_true == 0))
        fn = np.sum((y_pred_bin == 0) & (y_true == 1))
        
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


class GestorConfiguracion:
    """Gestor de configuraciones de entrenamiento."""
    
    @staticmethod
    def guardar_config(config: Dict[str, Any], ruta: str) -> None:
        """Guardar configuración en JSON."""
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuración guardada en: {ruta}")
    
    @staticmethod
    def cargar_config(ruta: str) -> Dict[str, Any]:
        """Cargar configuración de JSON."""
        with open(ruta, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuración cargada desde: {ruta}")
        return config
    
    @staticmethod
    def config_entrenamiento_default() -> Dict[str, Any]:
        """Configuración de entrenamiento por defecto."""
        return {
            'epochs': 100,
            'learning_rate': 0.01,
            'batch_size': 32,
            'l2_lambda': 0.0001,
            'tipo_perdida': 'mse',
            'verbose': 10
        }
