# Red Recurrente con GRU

Red Neuronal Recurrente (RNN) con Gated Recurrent Units (GRU).

## Características
- **GRU (Gated Recurrent Unit)**: Versión simplificada de LSTM
- **Memoria de secuencias**: Procesa datos temporales o secuenciales
- **Puertas de control**: 
  - Puerta de reset: controla qué información del pasado mantener
  - Puerta de actualización: controla la mezcla de información nueva y antigua
- **Aplicaciones**: Series temporales, procesamiento de lenguaje natural, análisis secuencial

## Conceptos
- Forward propagation a través del tiempo (BPTT)
- Estados ocultos recurrentes
- Softmax para clasificación multiclase

## Uso
```bash
python red_recurrente_gru.py
```

## Notas
Esta es una implementación educativa simplificada. En producción se recomienda usar frameworks como TensorFlow o PyTorch.
