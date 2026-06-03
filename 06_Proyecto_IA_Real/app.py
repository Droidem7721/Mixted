import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.panel import Panel
from main_gui import EditorCanvas
from core_engine import NeuralNetworkGraph

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
        
        btn_add_dense = Button(text='Añadir Capa Densa')
        btn_add_dense.bind(on_release=self.add_dense_layer)
        self.toolbar.add_widget(btn_add_dense)

        # Canvas del editor
        self.editor = EditorCanvas()
        self.add_widget(self.editor)

    def save_graph(self, instance):
        json_data = self.editor.graph.to_json()
        with open("network_save.json", "w") as f:
            f.write(json_data)
        print("Grafo guardado en network_save.json")

    def load_graph(self, instance):
        if os.path.exists("network_save.json"):
            with open("network_save.json", "r") as f:
                json_data = f.read()
            # Aquí se necesitaría lógica para limpiar el canvas y reconstruir visualmente
            print("Carga de grafo solicitada (lógica de reconstrucción visual pendiente)")
        else:
            print("No hay archivo de guardado.")

    def add_dense_layer(self, instance):
        self.editor.add_node_to_graph("Dense", "Nueva Capa", (100, 100))
        print("Nueva capa densa añadida.")

class NeuronalEditorApp(App):
    def build(self):
        return MainLayout()

if __name__ == '__main__':
    print("Iniciando Neuronal Editor...")
    # NeuronalEditorApp().run()
