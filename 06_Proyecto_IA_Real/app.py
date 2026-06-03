import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.panel import Panel
from main_gui import EditorCanvas
from core_engine import NeuralNetworkGraph

from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from tinygrad import Tensor

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(orientation='vertical', **kwargs)
        
        # Barra de herramientas superior
        self.toolbar = BoxLayout(size_hint_y=None, height=50, spacing=10, padding=5)
        self.add_widget(self.toolbar)
        
        btn_save = Button(text='Guardar')
        btn_save.bind(on_release=self.save_graph)
        self.toolbar.add_widget(btn_save)
        
        btn_load = Button(text='Cargar')
        btn_load.bind(on_release=self.load_graph)
        self.toolbar.add_widget(btn_load)
        
        btn_add_dense = Button(text='Capa Densa')
        btn_add_dense.bind(on_release=self.add_dense_layer)
        self.toolbar.add_widget(btn_add_dense)

        btn_compile = Button(text='Compilar (Tinygrad)', background_color=(0.2, 0.8, 0.2, 1))
        btn_compile.bind(on_release=self.compile_model)
        self.toolbar.add_widget(btn_compile)

        # Contenedor principal (Editor + Panel Propiedades)
        self.content = BoxLayout(orientation='horizontal')
        self.add_widget(self.content)

        # Canvas del editor
        self.editor = EditorCanvas()
        self.content.add_widget(self.editor)

        # Panel de propiedades (derecha)
        self.prop_panel = BoxLayout(orientation='vertical', size_hint_x=0.3, padding=10, spacing=10)
        with self.prop_panel.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.2, 0.2, 0.2, 1)
            self.prop_bg = Rectangle(pos=self.prop_panel.pos, size=self.prop_panel.size)
        self.prop_panel.bind(pos=self._update_prop_bg, size=self._update_prop_bg)
        self.content.add_widget(self.prop_panel)
        self.show_default_prop()

    def _update_prop_bg(self, *args):
        self.prop_bg.pos = self.prop_panel.pos
        self.prop_bg.size = self.prop_panel.size

    def show_default_prop(self):
        self.prop_panel.clear_widgets()
        self.prop_panel.add_widget(Label(text="Propiedades del Nodo", font_size='18sp', bold=True))
        self.prop_panel.add_widget(Label(text="Selecciona un nodo para editar"))

    def show_properties_panel(self, v_node):
        self.prop_panel.clear_widgets()
        self.prop_panel.add_widget(Label(text=f"Nodo: {v_node.node_type}", font_size='18sp', bold=True))
        
        # Editor de nombre
        self.prop_panel.add_widget(Label(text="Nombre:"))
        name_input = TextInput(text=v_node.node_obj.name, multiline=False)
        def update_name(instance, value): v_node.node_obj.name = value; v_node.label.text = value
        name_input.bind(text=update_name)
        self.prop_panel.add_widget(name_input)

        if v_node.node_type == "Dense":
            self.prop_panel.add_widget(Label(text="Unidades:"))
            units_input = TextInput(text=str(v_node.node_obj.parameters.get("units", 128)), multiline=False)
            def update_units(instance, value): 
                try: v_node.node_obj.parameters["units"] = int(value)
                except: pass
            units_input.bind(text=update_units)
            self.prop_panel.add_widget(units_input)

    def compile_model(self, instance):
        print("Compilando grafo a Tinygrad...")
        model = self.editor.graph.compiler.compile()
        # Prueba rápida de inferencia
        test_input = Tensor.randn(1, 784)
        output = model(test_input)
        print(f"Modelo compilado con éxito. Salida de prueba: {output.shape}")

    def save_graph(self, instance):
        json_data = self.editor.graph.to_json()
        with open("network_save.json", "w") as f:
            f.write(json_data)
        print("Grafo guardado en network_save.json")

    def load_graph(self, instance):
        if os.path.exists("network_save.json"):
            with open("network_save.json", "r") as f:
                json_data = f.read()
            print("Carga de grafo solicitada.")
        else:
            print("No hay archivo de guardado.")

    def add_dense_layer(self, instance):
        self.editor.add_node_to_graph("Dense", "Nueva Capa", (200, 200))
        print("Nueva capa densa añadida.")

class NeuronalEditorApp(App):
    def build(self):
        return MainLayout()

if __name__ == '__main__':
    print("Iniciando Neuronal Editor...")
    # NeuronalEditorApp().run()
