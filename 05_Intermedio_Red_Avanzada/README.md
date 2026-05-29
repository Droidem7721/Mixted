# Red Neuronal Avanzada

Red neuronal con técnicas avanzadas de entrenamiento y regularización.

## Características Avanzadas
- **Dropout**: Previene overfitting desactivando aleatoriamente neuronas
- **Momentum**: Acelera la convergencia del gradiente descendente
- **Regularización L2**: Penaliza pesos grandes para evitar overfitting
- **Inicialización He**: Inicialización inteligente de pesos para ReLU
- **Múltiples activaciones**: ReLU, Sigmoid, Tanh, Linear
- **Validación cruzada**: Separación de datos en train/val/test

## Uso
```bash
python red_avanzada.py
```

## Conceptos
- Forward propagation con dropout
- Backward propagation con momentum
- Entropía cruzada como función de pérdida
- Dataset no linealmente separable (Moons)
