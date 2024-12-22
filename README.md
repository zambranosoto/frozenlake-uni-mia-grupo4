# Maestría en Inteligencia Artificial
## UNI 2024-2
- **Curso:** Aprendizaje por Refuerzo
- **Integrantes Grupo 4:**
  - Acevedo Martínez, Ruddy.
  - Fuertes Malca, Ayrton.
  - Solís Almerco, Ricardo.
  - Vilcahuamán Dolorier, Dennis.
  - Zambrano Soto, José.

# Proyecto: Resolución del FrozenLake-v1 con Programación Dinámica


## Descripción del Proyecto
Este proyecto implementa una solución al entorno **FrozenLake-v1** de la librería **Gymnasium** utilizando el método de **Programación Dinámica** (*Value Iteration*). El objetivo es encontrar la política óptima que permita al agente alcanzar la meta en un lago helado sin caer en huecos.

FrozenLake es un entorno de aprendizaje por refuerzo en el que el agente se mueve sobre una cuadrícula representada como un lago helado. Las acciones del agente pueden ser determinísticas o estocásticas, lo que añade un desafío adicional al aprendizaje.

---

## Funcionalidad Principal

El programa incluye una **pantalla de selección de entorno** que permite al usuario elegir entre una cuadrícula de **4x4** o **8x8** para FrozenLake. Esta funcionalidad facilita explorar y probar diferentes configuraciones del entorno antes de iniciar el entrenamiento y la evaluación.

---

## Objetivos
- Implementar el método de **Iteración de Valores** (Value Iteration) para resolver el entorno FrozenLake-v1.
- Obtener la **política óptima** y probarla en múltiples episodios.
- Generar indicadores de desempeño, como el **número promedio de pasos**, la **tasa de éxito**, y el **promedio de recompensa por episodio**.
- Visualizar los resultados a través de gráficos y tablas para mostrar cómo el agente converge a la política óptima.

---

# 1. Modelado del Problema

## Definición del entorno
**FrozenLake-v1** es un entorno de aprendizaje por refuerzo donde un agente debe navegar por una cuadrícula helada para alcanzar un objetivo, evitando caer en huecos. El entorno puede ser de tamaño **4x4** o **8x8**, y está representado como un espacio de estados finito.
No hay un **"oponente"** explícito, pero la **estocasticidad** en las transiciones del entorno puede considerarse como un factor adverso.

## Modelado de estados y acciones

### - Estados:
Los estados están definidos por la posición del agente en la cuadrícula. En un entorno de tamaño n×n, hay n2 estados posibles. Cada celda puede ser:
- Inicio (S)
- Meta (G)
- Camino seguro (F)
- Hueco (H)

### - Acciones:
El agente puede realizar cuatro acciones:
- 0: Moverse hacia la izquierda
- 1: Moverse hacia abajo
- 2: Moverse hacia la derecha
- 3: Moverse hacia arriba

### - Transiciones:
El entorno es estocástico; es decir, las acciones tienen una probabilidad de no ejecutarse de manera exacta, desviando al agente hacia direcciones no deseadas.

# 2. Solución
## Método utilizado
Se utilizó Programación Dinámica mediante el algoritmo de Iteración de Valores (Value Iteration). Este método calcula la función de valor
V(s) para cada estado, actualizándola iterativamente hasta la convergencia. Una vez que
V(s) converge, se extrae la política óptima π(s) que maximiza la recompensa esperada.

### - Ecuación de Iteración de Valores:
$$
V(s) \leftarrow \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]
$$
> Donde:
> - P(s′∣s,a) es la probabilidad de transición al estado s′ desde s al realizar la acción a.
> - R(s,a,s′) es la recompensa por esa transición. 
> - 𝛾 es el factor de descuento.

## Implementación y ejecución
El algoritmo fue implementado en Python utilizando la librería **Gymnasium** para el entorno y herramientas adicionales para la visualización y evaluación. La política óptima fue evaluada en múltiples episodios para medir su desempeño.

# 3. Resultados
## Evaluación
- **Tasa de éxito:** El agente alcanzó la meta en un X% de los episodios evaluados. 
- **Recompensa promedio:** Recompensa acumulada obtenida en cada episodio, entre la cantidad de episodios evaluados.
- **Pasos promedio:** Cantidad de Pasos acumulados dio el agente en cada episidio hasta de llegar a un estado terminal: Hueco(H) o Meta(G), entre la cantidad de episodios evaluados.

## Visualizaciones
- **Convergencia:** Se generó una gráfica que muestra cómo la función de valor converge tras cada iteración.
![Convergencia](/results/convergence.png)


- **Mapa de calor de la política:** Ilustra las acciones preferidas en cada estado según la política óptima. Donde las acciones posibles son:
  - 0: Izquierda 
  - 1: Abajo 
  - 2: Derecha 
  - 3: Arriba

  ![Mapa de calor de la política](/results/policy_heatmap.png)


- **Recompensas promedio:** Presenta la evolución de las recompensas obtenidas por el agente durante los episodios de evaluación.
![Recompensas promedio](/results/average_rewards.png)


- **Mapa de Calor de los Valores de los Estados:** muestra el mapa de calor de los valores calculados para cada estado del entorno.
![Mapa de Calor de los Valores de los Estados](/results/value_heatmap.png)

# 4. Análisis
## Interpretación de los resultados
- **Convergencia:** La función de valor V(s) convergió rápidamente debido a la naturaleza pequeña del entorno.
- **Tasa de éxito:** Aunque la política es óptima, la estocasticidad del entorno afecta el desempeño, lo que explica que en algunos episodios el agente caiga en huecos.
- **Recompensa promedio:** Refleja el balance entre episodios exitosos y no exitosos.

## Comparaciones
Se probó el entorno con diferentes tamaños 4×4 y 8×8:
- En el tamaño **4×4**, la convergencia fue más rápida y el desempeño fue mejor debido a la menor complejidad del entorno.
- En el tamaño **8×8**, el agente requirió más iteraciones y episodios para obtener un desempeño consistente.

## Factores determinantes
- **Estocasticidad del entorno:** El principal desafío para el agente es la imprevisibilidad de las transiciones. 
- **Factor de descuento (𝛾):** Valores cercanos a 1 favorecen decisiones a largo plazo, mientras que valores menores priorizan recompensas inmediatas. 
- **Tamaño del entorno:** Entornos más grandes requieren más tiempo de cómputo y episodios de evaluación.

---

## Estructura del Proyecto

```plaintext
├── frozenlake/
│   ├── __init__.py         # Inicializa el paquete.
│   ├── environment.py      # Configuración y funciones relacionadas con el entorno.
│   ├── algorithms.py       # Implementaciones de programación dinámica (value iteration).
│   ├── utils.py            # Funciones de utilidad (guardar/cargar política, visualizaciones).
│   ├── evaluation.py       # Código para probar políticas y generar indicadores.
├── main.py                 # Punto de entrada principal para ejecutar el programa.
├── results/                # Carpeta donde se almacenan los gráficos y resultados.
└── README.md               # Documentación del proyecto. 
```

---

# 5. Instalación

## **Clonar el repositorio**:
   ```bash
   git clone https://github.com/zambranosoto/frozenlake-uni-mia-grupo4.git
   cd frozenlake-uni-mia-grupo4
   ```

## **Dependencia de librerías**:
  ```plaintext
  gymnasium==0.27.1       # Librería para crear y manejar entornos de aprendizaje por refuerzo
  pygame==2.1.3           # reemplaza con la versión que prefieras
  matplotlib==3.7.1       # Para graficar mapas de calor y curvas de convergencia
  numpy==1.25.0           # Operaciones numéricas
  tk==0.1.0               # Interfaz gráfica (tkinter)
  ```
