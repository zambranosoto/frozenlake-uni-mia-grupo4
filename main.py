# main.py
import tkinter as tk  # Librería para crear interfaces gráficas
from frozenlake.environment import create_environment  # Función para crear el entorno FrozenLake
from frozenlake.algorithms import value_iteration  # Algoritmo de Iteración de Valores
from frozenlake.utils import (  # Utilidades para guardar y visualizar resultados
    save_policy,
    plot_and_save_convergence,
    plot_and_save_policy_heatmap,
    plot_value_heatmap
)
from frozenlake.evaluation import evaluate_policy  # Función para evaluar la política óptima


def choose_environment_size():
    """
    Descripción:
        Interfaz gráfica para que el usuario elija el tamaño del entorno FrozenLake.
        Al seleccionar una opción, se cierra la ventana y se llama a la función principal
        para iniciar el entorno con el tamaño seleccionado.
    """
    def on_selection(map_size):
        """
        Maneja la selección del tamaño del mapa y cierra la ventana.
        Args:
            map_size (str): Tamaño del mapa ("4x4" o "8x8").
        """
        root.destroy()  # Cierra la ventana gráfica
        main_frozenlake(map_size)  # Llama a la función principal con el tamaño seleccionado

    # Crear la ventana principal
    root = tk.Tk()
    root.title("Seleccionar tamaño del entorno")  # Título de la ventana
    root.geometry("300x150")  # Dimensiones de la ventana

    # Etiqueta para el título
    label = tk.Label(root, text="Elige el tamaño del entorno", font=("Arial", 14))
    label.pack(pady=20)  # Agregar espacio alrededor de la etiqueta

    # Botón para seleccionar un entorno 4x4
    btn_4x4 = tk.Button(root, text="4x4", font=("Arial", 12), command=lambda: on_selection("4x4"))
    btn_4x4.pack(side=tk.LEFT, padx=20)  # Posición y margen del botón

    # Botón para seleccionar un entorno 8x8
    btn_8x8 = tk.Button(root, text="8x8", font=("Arial", 12), command=lambda: on_selection("8x8"))
    btn_8x8.pack(side=tk.RIGHT, padx=20)  # Posición y margen del botón

    # Mostrar la ventana y esperar la interacción del usuario
    root.mainloop()


def main_frozenlake(map_size):
    """
    Descripción:
        Función principal para entrenar y evaluar un agente en el entorno FrozenLake.
    Parámetros de entrada:
        map_size (str): Tamaño del entorno ("4x4" o "8x8").
    """
    # Crear el entorno FrozenLake
    env = create_environment(map_size=map_size, render_mode="human")

    # Entrenar el agente usando el algoritmo de Iteración de Valores
    policy, V, deltas = value_iteration(env)

    # Guardar la política óptima y generar gráficas
    env_shape = (4, 4) if map_size == "4x4" else (8, 8)  # Determinar dimensiones del entorno
    save_policy(policy)  # Guardar la política en un archivo
    plot_and_save_convergence(deltas)  # Guardar gráfica de convergencia
    plot_and_save_policy_heatmap(policy, env_shape=env_shape)  # Guardar mapa de calor de la política
    plot_value_heatmap(V, env_shape=env_shape)  # Guardar mapa de calor de la función de valor

    # Evaluar la política óptima en el entorno
    success_rate, average_reward, average_steps = evaluate_policy(
        env, policy, n_episodes=10, render=True
    )

    # Imprimir resultados de evaluación
    print(f"\nTasa de Éxito: {success_rate:.2f}%")
    print(f"Recompensa Promedio: {average_reward:.2f}")
    print(f"Pasos Promedio: {average_steps:.2f}")


if __name__ == "__main__":
    """
    Descripción:
        Punto de entrada principal del programa.
        Muestra la interfaz para elegir el tamaño del entorno y ejecuta el resto del código.
    """
    choose_environment_size()
