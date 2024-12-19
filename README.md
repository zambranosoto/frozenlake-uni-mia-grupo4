# Maestría en Inteligencia Artificial
## UNI 2024-2
- **Curso:** Aprendizaje por Refuerzo
- **Integrantes Grupo 4:**
  - Acevedo Martínez, Ruddy.
  - Fuertes Malca, Ayrton.
  - Solíz Almerco, Ricardo.
  - Vilcahuamán Dolorier, Dennis.
  - Zamrano Soto, José.

# Proyecto FrozenLake-v1


## Descripción del Proyecto
Este proyecto implementa una solución al entorno **FrozenLake-v1** de la librería **Gymnasium** utilizando el método de **Programación Dinámica** (*Value Iteration*). El objetivo es encontrar la política óptima que permita al agente alcanzar la meta en un lago helado sin caer en huecos.

FrozenLake es un entorno de aprendizaje por refuerzo en el que el agente se mueve sobre una cuadrícula representada como un lago helado. Las acciones del agente pueden ser determinísticas o estocásticas, lo que añade un desafío adicional al aprendizaje.

---

## Objetivos
- Implementar el método de **Iteración de Valores** (Value Iteration) para resolver el entorno FrozenLake-v1.
- Obtener la **política óptima** y probarla en múltiples episodios.
- Generar indicadores de desempeño, como el **número promedio de pasos**, la **tasa de éxito**, y el **promedio de recompensa por episodio**.
- Visualizar los resultados a través de gráficos y tablas para mostrar cómo el agente converge a la política óptima.

---

## Estructura del Proyecto

- ├── frozenlake/
- │ ├── init.py # Inicializa el paquete. 
- │ ├── environment.py # Configuración y funciones relacionadas con el entorno. 
- │ ├── algorithms.py # Implementaciones de programación dinámica (value iteration). 
- │ ├── utils.py # Funciones de utilidad (guardar/cargar política, visualizaciones). 
- │ ├── evaluation.py # Código para probar políticas y generar indicadores. 
- ├── main.py # Punto de entrada principal para ejecutar el programa. 
- └── README.md # Documentación del proyecto.
  - ├── results/  

---

## Instalación

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/zambranosoto/frozenlake-uni-mia-grupo4.git
   cd frozenlake-uni-mia-grupo4

## Resultados Generados:

- **Gráfica de convergencia:** results/convergence.png
  - Esta gráfica muestra cómo disminuye el valor máximo de cambio (delta) en la función de valores con cada iteración, hasta alcanzar la convergencia. 
    - El eje X representa el número de iteraciones. 
    - El eje Y muestra el valor máximo de cambio (delta).


- **Mapa de calor de la política:** results/policy_heatmap.png
  - Este mapa de calor ilustra la política óptima aprendida. 
  - Cada celda representa la acción preferida por el agente en un estado dado.ç
  - Acciones posibles:
    - 0: Izquierda 
    - 1: Abajo 
    - 2: Derecha 
    - 3: Arriba


- **Recompensas promedio:** results/average_rewards.png
  - Esta gráfica muestra el promedio de recompensas obtenidas por el agente a lo largo de múltiples episodios de prueba. 
    - El eje X representa los episodios. 
    - El eje Y muestra la recompensa promedio.


- **Métricas:** results/metrics.txt
  - Las métricas obtenidas durante la evaluación se guardan en el archivo results/metrics.txt.
  - Ejemplo de Métricas de Evaluación:
    - Tasa de Éxito: 100.00% 
    - Pasos Promedio: 12.34 
    - Recompensa Promedio: 0.90

