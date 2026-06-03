# Diseño de Arquitectura del Núcleo: Motor de Grafo Neuronal y Modelo de Datos

## 1. Introducción

Este documento detalla el diseño de la arquitectura del núcleo para el "Neuronal Editor", centrándose en el motor de grafo neuronal y el modelo de datos. El objetivo es crear una representación de red neuronal que sea flexible, extensible y fácil de serializar/deserializar, manteniendo la ligereza y la compatibilidad con el funcionamiento offline.

## 2. Componentes Clave

La arquitectura del núcleo se dividirá en los siguientes componentes principales:

*   **Nodos (Nodes)**: Representan las operaciones o capas de la red neuronal.
*   **Conexiones (Edges/Connections)**: Representan el flujo de datos entre nodos.
*   **Grafo de Red Neuronal (Neural Network Graph)**: La estructura completa que contiene nodos y conexiones.
*   **Serialización/Deserialización**: Mecanismos para guardar y cargar el grafo de la red.

## 3. Modelo de Datos

### 3.1. Clase `Node`

La clase `Node` será la abstracción base para todas las operaciones en la red neuronal. Cada nodo tendrá un tipo, un identificador único, parámetros configurables y puertos de entrada/salida.

| Atributo        | Tipo       | Descripción                                                                 |
| :-------------- | :--------- | :-------------------------------------------------------------------------- |
| `id`            | `str`      | Identificador único del nodo (UUID).                                        |
| `type`          | `str`      | Tipo de nodo (e.g., "Input", "Dense", "Activation", "Output").          |
| `name`          | `str`      | Nombre legible por humanos del nodo (opcional).                             |
| `parameters`    | `dict`     | Diccionario de parámetros específicos del nodo (e.g., `units` para `Dense`).|
| `input_ports`   | `list[str]`| Lista de identificadores de puertos de entrada.                             |
| `output_ports`  | `list[str]`| Lista de identificadores de puertos de salida.                              |
| `position`      | `tuple`    | Coordenadas (x, y) en el lienzo para la representación gráfica.             |

**Ejemplo de `Node` (JSON):**

```json
{
    "id": "node_123",
    "type": "Dense",
    "name": "Capa Densa 1",
    "parameters": {
        "units": 64,
        "activation": "relu"
    },
    "input_ports": ["in"],
    "output_ports": ["out"],
    "position": [100, 150]
}
```

### 3.2. Clase `Connection`

La clase `Connection` representará el flujo de datos entre un puerto de salida de un nodo y un puerto de entrada de otro nodo.

| Atributo            | Tipo   | Descripción                                                                 |
| :------------------ | :----- | :-------------------------------------------------------------------------- |
| `id`                | `str`  | Identificador único de la conexión (UUID).                                  |
| `source_node_id`    | `str`  | ID del nodo de origen.                                                      |
| `source_port_id`    | `str`  | ID del puerto de salida del nodo de origen.                                 |
| `target_node_id`    | `str`  | ID del nodo de destino.                                                     |
| `target_port_id`    | `str`  | ID del puerto de entrada del nodo de destino.                               |

**Ejemplo de `Connection` (JSON):**

```json
{
    "id": "conn_456",
    "source_node_id": "node_123",
    "source_port_id": "out",
    "target_node_id": "node_789",
    "target_port_id": "in"
}
```

### 3.3. Clase `NeuralNetworkGraph`

La clase `NeuralNetworkGraph` encapsulará la colección de nodos y conexiones, representando la red neuronal completa.

| Atributo    | Tipo                   | Descripción                                                     |
| :---------- | :--------------------- | :-------------------------------------------------------------- |
| `nodes`     | `list[Node]`           | Lista de todos los nodos en la red.                             |
| `connections` | `list[Connection]`     | Lista de todas las conexiones en la red.                        |

**Ejemplo de `NeuralNetworkGraph` (JSON):**

```json
{
    "nodes": [
        { ... node_123 ... },
        { ... node_789 ... }
    ],
    "connections": [
        { ... conn_456 ... }
    ]
}
```

## 4. Motor de Grafo Neuronal (Integración con Tinygrad)

El `NeuralNetworkGraph` servirá como la representación abstracta de la red. Para la ejecución y entrenamiento, esta representación se traducirá a un modelo de Tinygrad.

### 4.1. Traducción de Nodos a Capas de Tinygrad

Cada `Node` en el grafo se mapeará a una capa o operación de Tinygrad. Se necesitará un mecanismo para:

*   **Mapeo de Tipos**: Convertir `node.type` (e.g., "Dense") a la clase de capa correspondiente en Tinygrad (e.g., `tinygrad.nn.Linear`).
*   **Mapeo de Parámetros**: Pasar `node.parameters` a los constructores de las capas de Tinygrad.

### 4.2. Construcción del Modelo Secuencial/Funcional

Dado que Tinygrad es flexible, se puede construir el modelo de forma secuencial (para redes simples) o funcional (para redes más complejas con bifurcaciones). El `NeuralNetworkGraph` permitirá la construcción de un grafo computacional dinámico en Tinygrad, donde las conexiones definen el flujo de datos.

### 4.3. `GraphCompiler` (Propuesto)

Se podría introducir una clase `GraphCompiler` que tome un `NeuralNetworkGraph` y genere un modelo ejecutable de Tinygrad. Este compilador se encargaría de:

1.  **Validación del Grafo**: Asegurar que el grafo es válido (sin ciclos, puertos conectados correctamente, etc.).
2.  **Orden Topológico**: Determinar el orden de ejecución de los nodos.
3.  **Instanciación de Capas**: Crear las instancias de las capas de Tinygrad basadas en los nodos.
4.  **Conexión de Capas**: Establecer las conexiones entre las capas de Tinygrad según las `Connection`s del grafo.

## 5. Serialización y Persistencia

Se utilizará el formato **JSON** para la serialización del `NeuralNetworkGraph` debido a su legibilidad, simplicidad y facilidad de integración con Python. Cada nodo y conexión se representará como un objeto JSON, y el grafo completo como un objeto JSON que contiene listas de nodos y conexiones.

### 5.1. Métodos de Serialización/Deserialización

La clase `NeuralNetworkGraph` incluirá métodos para:

*   `to_json()`: Convertir el grafo a una cadena JSON.
*   `from_json(json_string)`: Reconstruir el grafo a partir de una cadena JSON.

Esto permitirá guardar y cargar redes neuronales de forma persistente en el sistema de archivos local, cumpliendo con el requisito de funcionamiento offline.

## 6. Consideraciones para la Ligereza y Offline

*   **UUIDs**: Se utilizarán UUIDs para `id`s de nodos y conexiones para asegurar unicidad sin depender de un sistema de base de datos.
*   **Validación en Tiempo de Diseño**: La validación del grafo se realizará en la capa de lógica del editor para proporcionar retroalimentación inmediata al usuario y evitar errores en tiempo de ejecución del motor de Tinygrad.
*   **Manejo de Errores**: Se implementará un manejo robusto de errores durante la construcción del modelo de Tinygrad para informar al usuario sobre configuraciones de red inválidas.

## 7. Próximos Pasos

La siguiente fase implicará la implementación de estas clases y la creación de un prototipo básico del `GraphCompiler` para demostrar la traducción de un grafo simple a un modelo de Tinygrad ejecutable.
