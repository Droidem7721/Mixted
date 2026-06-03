# Neuronal Editor - Proyecto IA Real

Este proyecto es un editor gráfico nativo para la creación y entrenamiento de redes neuronales, diseñado para ser ligero, offline y multiplataforma (Windows, Linux, macOS y Android 7+).

## Características Principales

- **Interfaz Gráfica Intuitiva**: Basada en Kivy, permite arrastrar nodos y conectar capas visualmente con curvas de Bezier.
- **Motor Neuronal Potente y Ligero**: Utiliza **Tinygrad**, un framework minimalista que permite inferencia y entrenamiento eficiente.
- **Compilador de Grafos**: Traduce automáticamente el diseño visual a un modelo ejecutable de Tinygrad.
- **Panel de Propiedades Dinámico**: Permite ajustar parámetros como el número de unidades en capas densas en tiempo real.
- **Totalmente Offline**: No requiere conexión a internet para funcionar.
- **Serialización JSON**: Guarda y carga tus modelos de forma persistente.

## Estructura del Proyecto

- `app.py`: Punto de entrada principal de la aplicación.
- `main_gui.py`: Implementación de la interfaz de usuario y el lienzo interactivo.
- `core_engine.py`: Lógica del grafo neuronal, serialización y compilación a Tinygrad.
- `ARCHITECTURE.md`: Documentación detallada de la arquitectura del sistema.

## Requisitos

- Python 3.7+
- Kivy
- Tinygrad

## Instalación

```bash
pip install kivy tinygrad
python app.py
```

## Uso

1. Ejecuta `app.py`.
2. Añade capas usando los botones de la barra de herramientas.
3. Arrastra los nodos para organizar tu red.
4. Haz doble clic en un nodo para editar sus propiedades en el panel derecho.
5. Haz clic en **Compilar (Tinygrad)** para generar el modelo y realizar una prueba de inferencia.
6. Guarda tu progreso con el botón **Guardar**.
