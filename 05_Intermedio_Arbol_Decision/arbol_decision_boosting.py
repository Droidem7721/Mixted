import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ArbolDecision:
    """Árbol de Decisión para clasificación."""
    
    class Nodo:
        def __init__(self, caracteristica=None, valor_split=None, izq=None, der=None, clase=None):
            self.caracteristica = caracteristica  # Índice de característica para split
            self.valor_split = valor_split        # Valor de split
            self.izq = izq                        # Subárbol izquierdo
            self.der = der                        # Subárbol derecho
            self.clase = clase                    # Clase si es hoja
    
    def __init__(self, max_profundidad=5, min_muestras_split=5):
        self.max_profundidad = max_profundidad
        self.min_muestras_split = min_muestras_split
        self.raiz = None
    
    def entropia(self, y):
        """Calcular entropía de Shanon."""
        _, conteos = np.unique(y, return_counts=True)
        probabilidades = conteos / len(y)
        return -np.sum(probabilidades * np.log2(probabilidades + 1e-8))
    
    def ganancia_informacion(self, padre, izq, der):
        """Calcular ganancia de información."""
        n = len(padre)
        n_izq, n_der = len(izq), len(der)
        
        if n_izq == 0 or n_der == 0:
            return 0
        
        entropia_padre = self.entropia(padre)
        entropia_izq = self.entropia(izq)
        entropia_der = self.entropia(der)
        
        entropia_hijos = (n_izq / n) * entropia_izq + (n_der / n) * entropia_der
        ganancia = entropia_padre - entropia_hijos
        
        return ganancia
    
    def mejor_split(self, X, y):
        """Encontrar el mejor split."""
        mejor_ganancia = -1
        mejor_caracteristica = None
        mejor_valor = None
        
        for caracteristica in range(X.shape[1]):
            valores = np.unique(X[:, caracteristica])
            
            for valor in valores:
                izq_idx = X[:, caracteristica] <= valor
                der_idx = ~izq_idx
                
                if np.sum(izq_idx) == 0 or np.sum(der_idx) == 0:
                    continue
                
                ganancia = self.ganancia_informacion(y, y[izq_idx], y[der_idx])
                
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_caracteristica = caracteristica
                    mejor_valor = valor
        
        return mejor_caracteristica, mejor_valor
    
    def construir_arbol(self, X, y, profundidad=0):
        """Construir árbol recursivamente."""
        n_muestras = len(y)
        n_clases = len(np.unique(y))
        
        # Condiciones de parada
        if (profundidad >= self.max_profundidad or
            n_muestras < self.min_muestras_split or
            n_clases == 1):
            clase_mayoritaria = np.bincount(y).argmax()
            return self.Nodo(clase=clase_mayoritaria)
        
        # Encontrar mejor split
        caracteristica, valor = self.mejor_split(X, y)
        
        if caracteristica is None:
            clase_mayoritaria = np.bincount(y).argmax()
            return self.Nodo(clase=clase_mayoritaria)
        
        # Split
        izq_idx = X[:, caracteristica] <= valor
        der_idx = ~izq_idx
        
        # Construir subárboles
        izq = self.construir_arbol(X[izq_idx], y[izq_idx], profundidad + 1)
        der = self.construir_arbol(X[der_idx], y[der_idx], profundidad + 1)
        
        return self.Nodo(caracteristica=caracteristica, valor_split=valor, izq=izq, der=der)
    
    def fit(self, X, y):
        """Entrenar árbol."""
        self.raiz = self.construir_arbol(X, y)
        return self
    
    def predecir_muestra(self, x, nodo):
        """Predecir para una muestra."""
        if nodo.clase is not None:
            return nodo.clase
        
        if x[nodo.caracteristica] <= nodo.valor_split:
            return self.predecir_muestra(x, nodo.izq)
        else:
            return self.predecir_muestra(x, nodo.der)
    
    def predict(self, X):
        """Predecir para múltiples muestras."""
        return np.array([self.predecir_muestra(x, self.raiz) for x in X])
    
    def profundidad_arbol(self, nodo=None):
        """Calcular profundidad del árbol."""
        if nodo is None:
            nodo = self.raiz
        
        if nodo.clase is not None:
            return 1
        
        return 1 + max(self.profundidad_arbol(nodo.izq), self.profundidad_arbol(nodo.der))
    
    def contar_nodos(self, nodo=None):
        """Contar número de nodos."""
        if nodo is None:
            nodo = self.raiz
        
        if nodo.clase is not None:
            return 1
        
        return 1 + self.contar_nodos(nodo.izq) + self.contar_nodos(nodo.der)


class BoostingArtesanal:
    """AdaBoost simple con árboles de decisión."""
    
    def __init__(self, n_estimadores=5, max_profundidad=3):
        self.n_estimadores = n_estimadores
        self.max_profundidad = max_profundidad
        self.modelos = []
        self.pesos_modelos = []
    
    def fit(self, X, y):
        """Entrenar AdaBoost."""
        n_muestras = len(y)
        pesos = np.ones(n_muestras) / n_muestras
        
        for i in range(self.n_estimadores):
            # Entrenar árbol con muestras ponderadas
            indices_muestreo = np.random.choice(n_muestras, n_muestras, p=pesos, replace=True)
            X_muestra = X[indices_muestreo]
            y_muestra = y[indices_muestreo]
            
            arbol = ArbolDecision(max_profundidad=self.max_profundidad)
            arbol.fit(X_muestra, y_muestra)
            
            # Calcular error
            predicciones = arbol.predict(X)
            error = np.sum(pesos[predicciones != y])
            
            if error > 0.5 or error == 0:
                alpha = 1 if error < 0.5 else 0.5
            else:
                alpha = 0.5 * np.log((1 - error) / error)
            
            # Actualizar pesos
            pesos = pesos * np.exp(-alpha * y * (2 * (predicciones == y) - 1))
            pesos = pesos / np.sum(pesos)
            
            self.modelos.append(arbol)
            self.pesos_modelos.append(alpha)
            
            print(f"Árbol {i+1}, Error: {error:.4f}, Alpha: {alpha:.4f}")
        
        return self
    
    def predict(self, X):
        """Predecir con ensemble."""
        predicciones = np.zeros((X.shape[0], 2))
        
        for arbol, alpha in zip(self.modelos, self.pesos_modelos):
            pred = arbol.predict(X)
            # Convertir a -1, 1
            pred_bipolar = 2 * pred - 1
            predicciones[:, 0] += alpha * pred_bipolar
        
        return (predicciones[:, 0] >= 0).astype(int)


if __name__ == "__main__":
    print("=" * 60)
    print("Árbol de Decisión y AdaBoost")
    print("=" * 60)
    
    # Generar datos
    X, y = make_classification(n_samples=300, n_features=4, n_classes=2, 
                              n_redundant=1, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar árbol simple
    print("\nEntrenando Árbol de Decisión...")
    arbol = ArbolDecision(max_profundidad=5)
    arbol.fit(X_train, y_train)
    
    pred_arbol = arbol.predict(X_test)
    acc_arbol = accuracy_score(y_test, pred_arbol)
    
    print(f"Profundidad del árbol: {arbol.profundidad_arbol()}")
    print(f"Número de nodos: {arbol.contar_nodos()}")
    print(f"Precisión (Árbol): {acc_arbol:.4f}")
    
    # Entrenar AdaBoost
    print("\nEntrenando AdaBoost...")
    boosting = BoostingArtesanal(n_estimadores=5, max_profundidad=3)
    boosting.fit(X_train, y_train)
    
    pred_boosting = boosting.predict(X_test)
    acc_boosting = accuracy_score(y_test, pred_boosting)
    
    print(f"\nPrecisión (AdaBoost): {acc_boosting:.4f}")
    
    # Métricas detalladas
    print("\nMétricas detalladas (AdaBoost):")
    print(f"Precisión: {precision_score(y_test, pred_boosting):.4f}")
    print(f"Recall: {recall_score(y_test, pred_boosting):.4f}")
    print(f"F1-Score: {f1_score(y_test, pred_boosting):.4f}")
