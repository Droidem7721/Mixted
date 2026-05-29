# Proyectos de Nivel Intermedio

Colección de proyectos avanzados sobre redes neuronales y machine learning implementados desde cero en Python.

## 📚 Proyectos Intermedios

### 1. **Red Neuronal Avanzada** (`05_Intermedio_Red_Avanzada/`)
- Red con técnicas avanzadas de entrenamiento
- **Dropout**: Prevención de overfitting
- **Momentum**: Optimización más rápida
- **Regularización L2**: Control de complejidad
- **Inicialización He**: Pesos iniciales inteligentes
- Ejemplo: Dataset Moons (clasificación no lineal)

### 2. **Red Recurrente con GRU** (`05_Intermedio_Red_Recurrente/`)
- Red Neuronal Recurrente (RNN) con Gated Recurrent Units
- **Memoria de secuencias**: Procesa datos temporales
- **Puertas de control**: Reset y actualización
- **BPTT**: Backpropagation a través del tiempo
- Ejemplo: Clasificación de secuencias

### 3. **K-Means Clustering** (`05_Intermedio_K_Means/`)
- Clustering no supervisado centroide
- **Método del Codo**: Encontrar k óptimo
- **Coeficiente de Silhueta**: Validar clustering
- **Iteración convergente**: Optimización local
- Ejemplo: Segmentación de datos

### 4. **Árbol de Decisión y AdaBoost** (`05_Intermedio_Arbol_Decision/`)
- Árbol de Decisión con ganancia de información
- **Ensemble Learning**: AdaBoost
- **Ponderación adaptativa**: Enfatiza casos difíciles
- **Clasificadores débiles**: Combinados para máximo rendimiento
- Ejemplo: Clasificación binaria

## 🚀 Requisitos

```bash
pip install numpy scikit-learn matplotlib
```

## 📖 Conceptos Cubiertos

✅ **Técnicas Avanzadas de Entrenamiento**
- Dropout y regularización
- Momentum y optimización adaptativa
- Validación y evaluación

✅ **Redes Recurrentes**
- Unidades recurrentes (GRU)
- Procesamiento secuencial
- Estados ocultos

✅ **Aprendizaje No Supervisado**
- Clustering centroide
- Métricas de validación
- Selección de k

✅ **Ensemble Learning**
- Métodos de boosting
- Árboles de decisión
- Combinación de modelos

## 🎯 Diferencias con Nivel Básico

| Aspecto | Básico | Intermedio |
|--------|--------|------------|
| Técnicas | Propagación forward/backward | Dropout, Momentum, L2, He |
| Redes | Densas simples | Recurrentes, Ensembles |
| Datos | Simples, separables | Complejos, no lineales |
| Validación | Train/test | Train/val/test |
| Métricas | Precisión | F1, Silhueta, Codo |
| Algoritmos | Básicos | Avanzados, No supervisado |

## 💡 Cómo Usar

Cada proyecto es independiente. Para ejecutar uno:

```bash
cd 05_Intermedio_Red_Avanzada
python red_avanzada.py
```

Cada proyecto genera gráficos y métricas de evaluación.

## 📝 Notas

- Todos implementados **desde cero** sin frameworks de deep learning
- Código comentado para facilitar aprendizaje
- Ejemplos prácticos con datos reales
- Recomendado tener conocimientos básicos de ML

---

**Autor:** Droidem7721  
**Año:** 2026  
**Nivel:** Intermedio 🟠
