import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class KMeans:
    """Algoritmo K-Means para clustering no supervisado."""
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroides = None
        self.etiquetas = None
        self.historial_inercia = []
    
    def inicializar_centroides(self, X):
        """Inicializar centroides aleatoriamente."""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def asignar_clusters(self, X):
        """Asignar cada punto al centroide más cercano."""
        distancias = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distancias[:, i] = np.linalg.norm(X - self.centroides[i], axis=1)
        return np.argmin(distancias, axis=1)
    
    def actualizar_centroides(self, X):
        """Actualizar centroides como promedio de puntos asignados."""
        nuevos_centroides = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            puntos_cluster = X[self.etiquetas == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides[i] = puntos_cluster.mean(axis=0)
            else:
                nuevos_centroides[i] = self.centroides[i]
        return nuevos_centroides
    
    def calcular_inercia(self, X):
        """Calcular inercia (suma de distancias dentro de clusters)."""
        inercia = 0
        for i in range(self.n_clusters):
            puntos_cluster = X[self.etiquetas == i]
            if len(puntos_cluster) > 0:
                inercia += np.sum(np.linalg.norm(puntos_cluster - self.centroides[i], axis=1) ** 2)
        return inercia
    
    def fit(self, X):
        """Entrenar K-Means."""
        self.centroides = self.inicializar_centroides(X)
        
        for iteration in range(self.max_iterations):
            # Asignar clusters
            self.etiquetas = self.asignar_clusters(X)
            
            # Guardar inercia
            inercia = self.calcular_inercia(X)
            self.historial_inercia.append(inercia)
            
            # Actualizar centroides
            nuevos_centroides = self.actualizar_centroides(X)
            
            # Verificar convergencia
            if np.allclose(self.centroides, nuevos_centroides):
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            self.centroides = nuevos_centroides
            
            if iteration % 10 == 0:
                print(f"Iteración {iteration}, Inercia: {inercia:.4f}")
        
        return self
    
    def predecir(self, X):
        """Predecir clusters para nuevos datos."""
        return self.asignar_clusters(X)
    
    def silhueta(self, X):
        """Calcular coeficiente de silhueta (0 a 1, mayor es mejor)."""
        silhuetas = []
        for i in range(X.shape[0]):
            punto = X[i]
            cluster_i = self.etiquetas[i]
            
            # a: distancia promedio dentro del cluster
            puntos_cluster = X[self.etiquetas == cluster_i]
            a = np.mean(np.linalg.norm(puntos_cluster - punto, axis=1))
            
            # b: distancia promedio al cluster más cercano
            b = np.inf
            for j in range(self.n_clusters):
                if j != cluster_i:
                    puntos_otros = X[self.etiquetas == j]
                    dist_media = np.mean(np.linalg.norm(puntos_otros - punto, axis=1))
                    b = min(b, dist_media)
            
            silhueta_i = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhuetas.append(silhueta_i)
        
        return np.mean(silhuetas)


class AnalizadorClusters:
    """Analizador para encontrar el número óptimo de clusters."""
    
    @staticmethod
    def metodo_codo(X, max_clusters=10):
        """Método del codo para encontrar clusters óptimo."""
        inercias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, max_iterations=100)
            kmeans.fit(X)
            inercias.append(kmeans.historial_inercia[-1])
            print(f"K={k}, Inercia: {inercias[-1]:.4f}")
        
        return inercias
    
    @staticmethod
    def metodo_silhueta(X, max_clusters=10):
        """Método de silhueta para validar clustering."""
        silhuetas = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, max_iterations=100)
            kmeans.fit(X)
            sil = kmeans.silhueta(X)
            silhuetas.append(sil)
            print(f"K={k}, Silhueta: {sil:.4f}")
        
        return silhuetas


if __name__ == "__main__":
    print("=" * 60)
    print("K-Means Clustering")
    print("=" * 60)
    
    # Generar datos
    X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # Encontrar número óptimo de clusters
    print("\nMétodo del Codo:")
    analizador = AnalizadorClusters()
    inercias = analizador.metodo_codo(X, max_clusters=8)
    
    print("\nMétodo de Silhueta:")
    silhuetas = analizador.metodo_silhueta(X, max_clusters=8)
    
    # Entrenar con k=4
    print("\n" + "="*60)
    print("Entrenando K-Means con k=4")
    print("="*60)
    kmeans = KMeans(n_clusters=4, max_iterations=100)
    kmeans.fit(X)
    
    # Evaluar
    sil_score = kmeans.silhueta(X)
    print(f"\nCoeficiente de Silhueta: {sil_score:.4f}")
    
    # Visualizar
    try:
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.etiquetas, cmap='viridis', alpha=0.6)
        plt.scatter(kmeans.centroides[:, 0], kmeans.centroides[:, 1], 
                   c='red', marker='X', s=200, label='Centroides')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.legend()
        plt.title('Clustering K-Means (k=4)')
        
        plt.subplot(1, 3, 2)
        plt.plot(range(1, 9), inercias, 'bo-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inercia')
        plt.title('Método del Codo')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(range(2, 9), silhuetas, 'go-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Coeficiente de Silhueta')
        plt.title('Método de Silhueta')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('06_kmeans_resultados.png')
        print("\nGráfico guardado como '06_kmeans_resultados.png'")
    except Exception as e:
        print(f"No se pudo guardar el gráfico: {e}")
