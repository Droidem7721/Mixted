"""Script de ejemplo completo para usar la red neuronal personalizable."""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from red_personalizable import RedNeuronalPersonalizable
from utilidades import (
    PreprocesadorDatos,
    GeneradorConfiguracion,
    EvaluadorMetricas,
    GestorConfiguracion
)


def ejemplo_1_clasificacion_binaria():
    """
    Ejemplo 1: Clasificación binaria simple.
    Caso de uso: Detección de spam, predicción de abandono, etc.
    """
    print("\n" + "="*70)
    print("EJEMPLO 1: Clasificación Binaria")
    print("="*70)
    
    # Generar datos
    X, y = make_classification(n_samples=300, n_features=4, n_classes=2, 
                              n_redundant=1, random_state=42)
    y = y.reshape(-1, 1).astype(float)
    
    # Preprocesat
    preprocesador = PreprocesadorDatos()
    X = preprocesador.estandarizar(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Configurar red
    config = GeneradorConfiguracion.simple()
    red = RedNeuronalPersonalizable(config)
    red.construir(dim_entrada=4)
    print(red.obtener_resumen())
    
    # Entrenar
    historial = red.entrenar(
        X_train, y_train, X_val, y_val,
        epochs=50, learning_rate=0.1, batch_size=16,
        tipo_perdida='entropia_cruzada', verbose=10
    )
    
    # Evaluar
    predicciones = red.predecir(X_test)
    precision = EvaluadorMetricas.precision(y_test, predicciones)
    cm = EvaluadorMetricas.confusion_matrix(y_test, predicciones)
    
    print(f"\nPrecisión en prueba: {precision:.4f}")
    print(f"Matriz de confusión: {cm}")
    
    return red, historial


def ejemplo_2_multiclase():
    """
    Ejemplo 2: Clasificación multiclase.
    Caso de uso: Clasificación de iris, reconocimiento de dígitos, etc.
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: Clasificación Multiclase (Iris)")
    print("="*70)
    
    # Cargar datos
    iris = load_iris()
    X, y = iris.data, iris.target
    y_onehot = np.eye(3)[y]  # One-hot encoding
    
    # Preprocesat
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Configurar (nota: usar softmax en salida para multiclase)
    config = [
        {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
        {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
        {'tipo': 'densa', 'neuronas': 3, 'activacion': 'sigmoid'}  # 3 clases
    ]
    
    red = RedNeuronalPersonalizable(config)
    red.construir(dim_entrada=4)
    print(red.obtener_resumen())
    
    # Entrenar
    red.entrenar(
        X_train, y_train, X_val, y_val,
        epochs=50, learning_rate=0.1, batch_size=8,
        tipo_perdida='entropia_cruzada', verbose=10
    )
    
    # Evaluar
    predicciones = red.predecir(X_test)
    precision = np.mean(np.argmax(predicciones, axis=1) == np.argmax(y_test, axis=1))
    print(f"\nPrecisión en prueba: {precision:.4f}")
    
    return red


def ejemplo_3_regresion():
    """
    Ejemplo 3: Regresión.
    Caso de uso: Predicción de precios, temperatura, ventas, etc.
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: Regresión")
    print("="*70)
    
    # Generar datos
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = (np.sin(X) + np.random.randn(200, 1) * 0.1).reshape(-1, 1)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Configurar (salida linear para regresión)
    config = GeneradorConfiguracion.regresion()
    red = RedNeuronalPersonalizable(config)
    red.construir(dim_entrada=1)
    print(red.obtener_resumen())
    
    # Entrenar
    red.entrenar(
        X_train, y_train, X_val, y_val,
        epochs=100, learning_rate=0.1, batch_size=16,
        tipo_perdida='mse', verbose=20
    )
    
    # Evaluar
    predicciones = red.predecir(X_test)
    rmse = EvaluadorMetricas.rmse(y_test, predicciones)
    mae = EvaluadorMetricas.mae(y_test, predicciones)
    
    print(f"\nRMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    return red, X, y, X_test, predicciones


def ejemplo_4_personalizacion_completa():
    """
    Ejemplo 4: Personalización completa con configuración guardada.
    Demuestra cómo guardar, cargar y reutilizar modelos.
    """
    print("\n" + "="*70)
    print("EJEMPLO 4: Personalización Completa")
    print("="*70)
    
    # Generar datos
    X, y = make_classification(n_samples=400, n_features=8, n_classes=2, 
                              n_redundant=2, random_state=42)
    y = y.reshape(-1, 1).astype(float)
    
    # Preprocesat
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Configuración personalizada
    config_arquitectura = [
        {'tipo': 'densa', 'neuronas': 32, 'activacion': 'relu', 'dropout': 0.3, 'nombre': 'entrada'},
        {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2, 'nombre': 'oculta1'},
        {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1, 'nombre': 'oculta2'},
        {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid', 'nombre': 'salida'}
    ]
    
    # Configuración de entrenamiento
    config_entrenamiento = {
        'epochs': 50,
        'learning_rate': 0.05,
        'batch_size': 32,
        'l2_lambda': 0.0001,
        'tipo_perdida': 'entropia_cruzada'
    }
    
    # Guardar configuración
    GestorConfiguracion.guardar_config(
        {'arquitectura': config_arquitectura, 'entrenamiento': config_entrenamiento},
        'configs/modelo_personalizado.json'
    )
    
    # Crear y entrenar red
    red = RedNeuronalPersonalizable(config_arquitectura)
    red.construir(dim_entrada=8)
    print(red.obtener_resumen())
    
    red.entrenar(
        X_train, y_train, X_val, y_val,
        epochs=config_entrenamiento['epochs'],
        learning_rate=config_entrenamiento['learning_rate'],
        batch_size=config_entrenamiento['batch_size'],
        l2_lambda=config_entrenamiento['l2_lambda'],
        tipo_perdida=config_entrenamiento['tipo_perdida'],
        verbose=10
    )
    
    # Guardar modelo
    ruta_modelo = red.guardar()
    
    # Cargar modelo y hacer predicciones
    red_cargada = RedNeuronalPersonalizable.cargar(ruta_modelo)
    predicciones = red_cargada.predecir(X_test)
    precision = EvaluadorMetricas.precision(y_test, predicciones)
    
    print(f"\nPrecisión después de cargar: {precision:.4f}")
    
    return red_cargada


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  Red Neuronal Personalizable - Ejemplos de Uso  ".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Ejecutar ejemplos
    print("\n¿Cuál ejemplo deseas ejecutar?")
    print("1. Clasificación Binaria")
    print("2. Clasificación Multiclase (Iris)")
    print("3. Regresión")
    print("4. Personalización Completa")
    print("5. Todos")
    
    opcion = input("\nSelecciona (1-5): ").strip()
    
    if opcion == '1':
        ejemplo_1_clasificacion_binaria()
    elif opcion == '2':
        ejemplo_2_multiclase()
    elif opcion == '3':
        ejemplo_3_regresion()
    elif opcion == '4':
        ejemplo_4_personalizacion_completa()
    elif opcion == '5':
        ejemplo_1_clasificacion_binaria()
        ejemplo_2_multiclase()
        ejemplo_3_regresion()
        ejemplo_4_personalizacion_completa()
    else:
        print("Opción no válida")
