# Transformers y Redes Neuronales Avanzadas

Implementación avanzada de Transformers y Redes Neuronales de Grafos (GNN).

## 🚀 Características

### Transformer Simplificado
- **Mecanismo de Atención Multi-Cabeza**: Atención escalada puntual
- **Bloques de Decoder**: Auto-atención + Atención Cruzada + FFN
- **Normalización de Capa**: Layer Normalization
- **Embeddings Posicionales**: Codificación absoluta
- **Generación Autorreégresiva**: Muestreo con temperatura
- **Dropout**: Regularización

### Red Neuronal de Grafos (GNN)
- **Convolución de Grafo**: Agregación de vecinos
- **Normalización Simétrica**: D^-1/2 A D^-1/2
- **Múltiples Capas**: Capas de convolución apiladas
- **Clasificación de Nodos**: Tarea de clasificación semi-supervisada

## 📁 Arquitectura Transformer

```
[Entrada]
    ↓
[Embedding + Positional Encoding]
    ↓
[Decoder Block 1]
  │
  ├─ [Self-Attention]
  │
  ├─ [Cross-Attention]  
  │
  ├─ [Feed-Forward]
  │
  └─ [Layer Norm]
    ↓
[Decoder Block N]
    ↓
[Linear + Softmax]
    ↓
[Salida]
```

## 📁 Arquitectura GNN

```
Nodos + Matriz de Adyacencia
    ↓
[Convolución 1: A X W]
    ↓
[ReLU]
    ↓
[Convolución 2: A X W]
    ↓
[Embeddings de Nodos]
    ↓
[Clasificación]
```

## 📚 Casos de Uso

### Transformers
- Traducción automática
- Generación de texto
- Modelado de lenguaje
- Resumen de textos

### GNN
- Clasificación de moléculas
- Análisis de redes sociales
- Sistemas de recomendación
- Análisis de conocimiento

## 🧰 Conceptos Clave

- **Atención**: Mecanismo para ponderar importancia de tokens
- **Multi-Head**: Múltiples espacios de atención simultáneamente
- **Mascara Causal**: Prevenir que el modelo mire hacia el futuro
- **Agregación de Grafos**: Combinar información de nodos vecinos
- **Normalización Simétrica**: Asegurar convergencia en grafos
