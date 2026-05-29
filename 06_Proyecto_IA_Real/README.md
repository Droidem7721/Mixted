# Red Neuronal Personalizable - Proyecto Real

Proyecto completo de una Red Neuronal flexible y reutilizable para múltiples propósitos.

## 🎯 Características Principales

### ✨ Versatilidad
- **Arquitectura flexible**: Define capas, activaciones y dropouts
- **Múltiples tipos de pérdida**: MSE, Entropía Cruzada
- **Fácil personalización**: Cambia configuración para cualquier problema

### 🔧 Funcionalidades
- Forward/Backward propagation completo
- Dropout para regularización
- Momentum para optimización
- Regularización L2
- Inicialización inteligente (He, Xavier)
- Guardado/Carga de modelos
- Historial de entrenamiento

### 📊 Casos de Uso
1. **Clasificación Binaria**: Detección de spam, predicción de abandono
2. **Clasificación Multiclase**: Reconocimiento de dígitos, clasificación de iris
3. **Regresión**: Predicción de precios, temperaturas, ventas
4. **Cualquier otro problema** que se pueda modelar con redes neuronales

## 📁 Archivos

- `red_personalizable.py`: Clase principal con red neuronal
- `utilidades.py`: Utilidades para preprocesamiento, configuración y métricas
- `ejemplos.py`: 4 ejemplos completos de uso
- `README.md`: Este archivo

## 🚀 Uso Rápido

### Instalación
```bash
pip install numpy scikit-learn matplotlib
```

### Uso Básico

```python
from red_personalizable import RedNeuronalPersonalizable
from utilidades import GeneradorConfiguracion

# 1. Define arquitectura
config = [
    {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.2},
    {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.1},
    {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid'}
]

# 2. Crea red
red = RedNeuronalPersonalizable(config)
red.construir(dim_entrada=4)

# 3. Entrena
red.entrenar(X_train, y_train, X_val, y_val, 
             epochs=50, learning_rate=0.1)

# 4. Predice
predicciones = red.predecir(X_test)

# 5. Guarda
red.guardar('mi_modelo.pkl')
```

## 📚 Configuraciones Predefinidas

```python
from utilidades import GeneradorConfiguracion

# Simple: 1 capa oculta
config = GeneradorConfiguracion.simple()

# Moderada: 2 capas ocultas
config = GeneradorConfiguracion.moderada()

# Profunda: 3 capas ocultas
config = GeneradorConfiguracion.profunda()

# Multiclase: para N clases
config = GeneradorConfiguracion.multiclase(n_clases=5)

# Regresión: para predicción continua
config = GeneradorConfiguracion.regresion()
```

## 🔍 Métricas de Evaluación

```python
from utilidades import EvaluadorMetricas

# Precisión
prec = EvaluadorMetricas.precision(y_true, y_pred)

# Error medio absoluto
mae = EvaluadorMetricas.mae(y_true, y_pred)

# Raíz del error cuadrático medio
rmse = EvaluadorMetricas.rmse(y_true, y_pred)

# Matriz de confusión
cm = EvaluadorMetricas.confusion_matrix(y_true, y_pred)
```

## 📝 Ejemplos Completos

### Ejecutar ejemplos
```bash
python ejemplos.py
```

Luego selecciona:
- `1`: Clasificación Binaria
- `2`: Clasificación Multiclase (Iris)
- `3`: Regresión
- `4`: Personalización Completa
- `5`: Todos

### Ejemplo: Clasificación Binaria

```python
from ejemplos import ejemplo_1_clasificacion_binaria

red, historial = ejemplo_1_clasificacion_binaria()
```

### Ejemplo: Regresión

```python
from ejemplos import ejemplo_3_regresion

red, X, y, X_test, predicciones = ejemplo_3_regresion()
```

## 🛠️ Personalización Avanzada

### Crear arquitectura personalizada

```python
config = [
    {'tipo': 'densa', 'neuronas': 64, 'activacion': 'relu', 'dropout': 0.3, 'nombre': 'entrada'},
    {'tipo': 'densa', 'neuronas': 32, 'activacion': 'relu', 'dropout': 0.2, 'nombre': 'oculta1'},
    {'tipo': 'densa', 'neuronas': 16, 'activacion': 'relu', 'dropout': 0.1, 'nombre': 'oculta2'},
    {'tipo': 'densa', 'neuronas': 8, 'activacion': 'relu', 'dropout': 0.05, 'nombre': 'oculta3'},
    {'tipo': 'densa', 'neuronas': 1, 'activacion': 'sigmoid', 'nombre': 'salida'}
]

red = RedNeuronalPersonalizable(config)
red.construir(dim_entrada=20)
```

### Entrenamiento con parámetros personalizados

```python
red.entrenar(
    X_train, y_train,
    X_val, y_val,
    epochs=200,
    learning_rate=0.01,
    batch_size=64,
    l2_lambda=0.0001,
    tipo_perdida='entropia_cruzada',
    verbose=5
)
```

## 💾 Guardar y Cargar

```python
# Guardar
ruta = red.guardar('modelos/mi_modelo_v1.pkl')

# Cargar
red_cargada = RedNeuronalPersonalizable.cargar(ruta)

# Usar modelo cargado
predicciones = red_cargada.predecir(X_nuevo)
```

## 📊 Opciones de Activación

- `relu`: ReLU (Rectified Linear Unit) - Para capas ocultas
- `sigmoid`: Sigmoide - Para clasificación binaria
- `tanh`: Tangente Hiperbólica - Para normalizar [-1, 1]
- `linear`: Lineal - Para regresión
- `softmax`: (implementación futura) - Para multiclase

## ⚙️ Hiperparámetros Importantes

| Parámetro | Rango | Default | Efecto |
|-----------|-------|---------|--------|
| `learning_rate` | 0.001 - 0.1 | 0.01 | Tamaño de paso en optimización |
| `batch_size` | 8 - 256 | 32 | Muestras por actualización |
| `dropout` | 0.0 - 0.5 | 0.2 | Proporción de neuronas a desactivar |
| `l2_lambda` | 0.0 - 0.01 | 0.0 | Fuerza de regularización |
| `epochs` | 10 - 1000 | 100 | Ciclos completos de entrenamiento |
| `momentum` | 0.0 - 0.99 | 0.9 | Aceleración en gradiente descendente |

## 🎓 Conceptos de Aprendizaje

Esta implementación enseña:
- ✅ Forward propagation
- ✅ Backward propagation (Backpropagation)
- ✅ Descenso de gradiente con momentum
- ✅ Regularización (Dropout, L2)
- ✅ Validación y evaluación
- ✅ Persistencia de modelos
- ✅ Arquitecturas modulares

## 🔗 Adaptación para Nuevos Problemas

### Paso 1: Prepara tus datos
```python
from utilidades import PreprocesadorDatos

preprocesador = PreprocesadorDatos()
X_procesado = preprocesador.estandarizar(X)
```

### Paso 2: Elige configuración
```python
from utilidades import GeneradorConfiguracion

# Binaria
config = GeneradorConfiguracion.simple()
# O multiclase
config = GeneradorConfiguracion.multiclase(n_clases=10)
```

### Paso 3: Crea y entrena
```python
red = RedNeuronalPersonalizable(config)
red.construir(dim_entrada=X_procesado.shape[1])
red.entrenar(X_train, y_train, X_val, y_val, epochs=50)
```

### Paso 4: Evalúa y guarda
```python
resultados = red.evaluar(X_test, y_test)
red.guardar('mi_modelo_final.pkl')
```

## 🚨 Solución de Problemas

### Modelo no converge
- Reduce `learning_rate`
- Aumenta `epochs`
- Verifica preprocesamiento de datos

### Overfitting
- Aumenta `dropout`
- Aumenta `l2_lambda`
- Más datos de entrenamiento

### Underfitting
- Aumenta `learning_rate`
- Aumenta capas ocultas
- Reduce regularización

## 📚 Referencias y Recursos

- Implementación educativa desde cero
- Basada en conceptos fundamentales de deep learning
- Sin dependencias de frameworks (solo NumPy y scikit-learn)

## 📄 Licencia

Código educativo - Libre para usar y modificar

---

**Autor**: Droidem7721  
**Año**: 2026  
**Versión**: 1.0  
**Propósito**: Educativo y Proyectos Reales
