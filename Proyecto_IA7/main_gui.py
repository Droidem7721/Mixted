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
    selected = ListProperty([0, 0, 0, 0]) # Borde de selección

    def __init__(self, node_obj, **kwargs):
        super(VisualNode, self).__init__(**kwargs)
        self.node_id = node_obj.id
        self.node_type = node_obj.type
        self.node_obj = node_obj
        self.size = (140, 70)
        self.pos = node_obj.position
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        
        with self.canvas.before:
            self.bg_color = Color(*self.color)
            self.rect = Rectangle(pos=self.pos, size=self.size)
            self.border_color = Color(1, 1, 1, 0) # Invisible por defecto
            self.border = Line(rectangle=(self.x, self.y, self.width, self.height), width=2)
        
        self.label = Label(text=node_obj.name, center=self.center, color=(1, 1, 1, 1), font_size='14sp')
        self.add_widget(self.label)
        self.bind(pos=self.update_label)

    def update_canvas(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
        self.border.rectangle = (self.x, self.y, self.width, self.height)

    def update_label(self, *args):
        self.label.center = self.center

    def select(self):
        self.border_color.a = 1
        
    def deselect(self):
        self.border_color.a = 0

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_double_tap:
                # Abrir panel de propiedades (se manejará en el padre)
                self.parent.open_properties(self)
            touch.grab(self)
            self.parent.select_node(self)
            return True
        return super(VisualNode, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.pos = (self.pos[0] + touch.dx, self.pos[1] + touch.dy)
            self.node_obj.position = self.pos
            self.parent.update_connections()
            return True
        return super(VisualNode, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super(VisualNode, self).on_touch_up(touch)

class EditorCanvas(Widget):
    selected_node = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super(EditorCanvas, self).__init__(**kwargs)
        self.graph = NeuralNetworkGraph()
        self.visual_nodes = {}
        
        # Fondo oscuro para el editor
        with self.canvas.before:
            Color(0.15, 0.15, 0.15, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        # Nodos iniciales
        self.add_node_to_graph("Input", "Entrada", (100, 300))
        self.add_node_to_graph("Dense", "Capa Oculta", (300, 300))
        self.add_node_to_graph("Output", "Salida", (550, 300))
        
        node_ids = list(self.visual_nodes.keys())
        self.add_connection_to_graph(node_ids[0], node_ids[1])
        self.add_connection_to_graph(node_ids[1], node_ids[2])

    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def select_node(self, v_node):
        if self.selected_node:
            self.selected_node.deselect()
        self.selected_node = v_node
        v_node.select()

    def open_properties(self, v_node):
        # Esta función será llamada por la App principal para mostrar el panel
        if self.parent and hasattr(self.parent, 'show_properties_panel'):
            self.parent.show_properties_panel(v_node)

    def add_node_to_graph(self, node_type, name, pos):
        params = {"units": 128} if node_type == "Dense" else {}
        node_obj = Node(node_type, name, parameters=params, position=pos)
        self.graph.add_node(node_obj)
        v_node = VisualNode(node_obj)
        self.add_widget(v_node)
        self.visual_nodes[node_obj.id] = v_node
        return v_node

    def add_connection_to_graph(self, src_id, tgt_id):
        conn = Connection(src_id, "out", tgt_id, "in")
        self.graph.add_connection(conn)
        self.update_connections()

    def update_connections(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            for conn in self.graph.connections:
                src_node = self.visual_nodes.get(conn.source_node_id)
                tgt_node = self.visual_nodes.get(conn.target_node_id)
                if src_node and tgt_node:
                    Color(0.4, 0.4, 0.4, 1)
                    start_pos = (src_node.x + src_node.width, src_node.y + src_node.height / 2)
                    end_pos = (tgt_node.x, tgt_node.y + tgt_node.height / 2)
                    # Dibujar curva de Bezier para una apariencia más profesional
                    mid_x = (start_pos[0] + end_pos[0]) / 2
                    Line(bezier=[start_pos[0], start_pos[1], mid_x, start_pos[1], mid_x, end_pos[1], end_pos[0], end_pos[1]], width=2)

class NeuronalEditorApp(App):
    def build(self):
        return EditorCanvas()

if __name__ == '__main__':
    # Nota: En este entorno de sandbox no podemos ejecutar la GUI directamente
    # pero el código está listo para ser desplegado.
    print("Aplicación Kivy lista para ser ejecutada en un entorno con pantalla.")
    # NeuronalEditorApp().run()
