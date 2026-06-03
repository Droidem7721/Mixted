import json
import uuid

class Node:
    def __init__(self, node_type, name=None, parameters=None, position=(0, 0)):
        self.id = str(uuid.uuid4())
        self.type = node_type
        self.name = name or f"{node_type}_{self.id[:8]}"
        self.parameters = parameters or {}
        self.position = position
        self.input_ports = ["in"]
        self.output_ports = ["out"]

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "parameters": self.parameters,
            "position": self.position,
            "input_ports": self.input_ports,
            "output_ports": self.output_ports
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data["type"], data["name"], data["parameters"], data["position"])
        node.id = data["id"]
        node.input_ports = data["input_ports"]
        node.output_ports = data["output_ports"]
        return node

class Connection:
    def __init__(self, source_node_id, source_port_id, target_node_id, target_port_id):
        self.id = str(uuid.uuid4())
        self.source_node_id = source_node_id
        self.source_port_id = source_port_id
        self.target_node_id = target_node_id
        self.target_port_id = target_port_id

    def to_dict(self):
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "source_port_id": self.source_port_id,
            "target_node_id": self.target_node_id,
            "target_port_id": self.target_port_id
        }

    @classmethod
    def from_dict(cls, data):
        conn = cls(data["source_node_id"], data["source_port_id"], data["target_node_id"], data["target_port_id"])
        conn.id = data["id"]
        return conn

class NeuralNetworkGraph:
    def __init__(self):
        self.nodes = {}
        self.connections = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.connections = [c for c in self.connections if c.source_node_id != node_id and c.target_node_id != node_id]

    def add_connection(self, connection):
        self.connections.append(connection)

    def remove_connection(self, connection_id):
        self.connections = [c for c in self.connections if c.id != connection_id]

    def to_json(self):
        return json.dumps({
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections]
        }, indent=4)

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        graph = cls()
        for node_data in data["nodes"]:
            graph.add_node(Node.from_dict(node_data))
        for conn_data in data["connections"]:
            graph.add_connection(Connection.from_dict(conn_data))
        return graph

# Ejemplo de uso
if __name__ == "__main__":
    graph = NeuralNetworkGraph()
    
    # Crear nodos
    input_node = Node("Input", "Entrada", position=(50, 100))
    dense_node = Node("Dense", "Capa Densa 1", {"units": 32, "activation": "relu"}, position=(200, 100))
    output_node = Node("Output", "Salida", position=(350, 100))
    
    graph.add_node(input_node)
    graph.add_node(dense_node)
    graph.add_node(output_node)
    
    # Crear conexiones
    conn1 = Connection(input_node.id, "out", dense_node.id, "in")
    conn2 = Connection(dense_node.id, "out", output_node.id, "in")
    
    graph.add_connection(conn1)
    graph.add_connection(conn2)
    
    # Serializar
    json_output = graph.to_json()
    print("Grafo Serializado:")
    print(json_output)
    
    # Deserializar
    new_graph = NeuralNetworkGraph.from_json(json_output)
    print("\nGrafo Deserializado con éxito.")
    print(f"Nodos: {len(new_graph.nodes)}")
    print(f"Conexiones: {len(new_graph.connections)}")
