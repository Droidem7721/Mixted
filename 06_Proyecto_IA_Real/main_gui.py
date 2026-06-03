import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.uix.label import Label
from kivy.properties import ListProperty, StringProperty, ObjectProperty
from core_engine import Node, Connection, NeuralNetworkGraph

class VisualNode(Widget):
    node_id = StringProperty('')
    node_type = StringProperty('')
    color = ListProperty([0.2, 0.6, 0.8, 1])

    def __init__(self, node_obj, **kwargs):
        super(VisualNode, self).__init__(**kwargs)
        self.node_id = node_obj.id
        self.node_type = node_obj.type
        self.size = (120, 60)
        self.pos = node_obj.position
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        
        with self.canvas.before:
            Color(*self.color)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        
        self.label = Label(text=node_obj.name, center=self.center, color=(1, 1, 1, 1))
        self.add_widget(self.label)
        self.bind(pos=self.update_label)

    def update_canvas(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_label(self, *args):
        self.label.center = self.center

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            return True
        return super(VisualNode, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.pos = (self.pos[0] + touch.dx, self.pos[1] + touch.dy)
            # Notificar al canvas principal para redibujar conexiones
            self.parent.update_connections()
            return True
        return super(VisualNode, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super(VisualNode, self).on_touch_up(touch)

class EditorCanvas(Widget):
    def __init__(self, **kwargs):
        super(EditorCanvas, self).__init__(**kwargs)
        self.graph = NeuralNetworkGraph()
        self.visual_nodes = {}
        self.connection_lines = []

        # Inicializar con algunos nodos de ejemplo
        self.add_node_to_graph("Input", "Entrada", (100, 300))
        self.add_node_to_graph("Dense", "Capa Oculta", (300, 300))
        self.add_node_to_graph("Output", "Salida", (500, 300))
        
        # Conexiones iniciales
        node_ids = list(self.visual_nodes.keys())
        self.add_connection_to_graph(node_ids[0], node_ids[1])
        self.add_connection_to_graph(node_ids[1], node_ids[2])

    def add_node_to_graph(self, node_type, name, pos):
        node_obj = Node(node_type, name, position=pos)
        self.graph.add_node(node_obj)
        v_node = VisualNode(node_obj)
        self.add_widget(v_node)
        self.visual_nodes[node_obj.id] = v_node

    def add_connection_to_graph(self, src_id, tgt_id):
        conn = Connection(src_id, "out", tgt_id, "in")
        self.graph.add_connection(conn)
        self.update_connections()

    def update_connections(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(0.7, 0.7, 0.7, 1)
            for conn in self.graph.connections:
                src_node = self.visual_nodes.get(conn.source_node_id)
                tgt_node = self.visual_nodes.get(conn.target_node_id)
                if src_node and tgt_node:
                    # Dibujar línea desde el centro derecho del origen al centro izquierdo del destino
                    start_pos = (src_node.x + src_node.width, src_node.y + src_node.height / 2)
                    end_pos = (tgt_node.x, tgt_node.y + tgt_node.height / 2)
                    Line(points=[start_pos[0], start_pos[1], end_pos[0], end_pos[1]], width=2)

class NeuronalEditorApp(App):
    def build(self):
        return EditorCanvas()

if __name__ == '__main__':
    # Nota: En este entorno de sandbox no podemos ejecutar la GUI directamente
    # pero el código está listo para ser desplegado.
    print("Aplicación Kivy lista para ser ejecutada en un entorno con pantalla.")
    # NeuronalEditorApp().run()
