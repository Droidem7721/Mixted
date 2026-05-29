# K-Means Clustering

Algoritmo K-Means para clustering no supervisado (aprendizaje sin etiquetas).

## Características
- **Clustering centroide**: Agrupa datos en k clusters
- **Método del Codo**: Determina el número óptimo de clusters
- **Coeficiente de Silhueta**: Evalúa la calidad del clustering
- **Algoritmo iterativo**: Converge a mínimo local
- **Aplicaciones**: Segmentación de clientes, compresión de imágenes, análisis de datos

## Conceptos
- Inicialización de centroides
- Asignación de puntos a clusters
- Actualización iterativa de centroides
- Métricas de validación

## Uso
```bash
python kmeans_clustering.py
```

## Ventajas
- Simple e interpretable
- Eficiente computacionalmente
- Escalable a grandes datasets

## Desventajas
- Requiere especificar k de antemano
- Sensible a inicialización
- Asume clusters esféricos
