# main.py
import tkinter as tk
from frozenlake.environment import create_environment
from frozenlake.algorithms import value_iteration
from frozenlake.utils import save_policy, plot_and_save_convergence, plot_and_save_policy_heatmap, plot_value_heatmap
from frozenlake.evaluation import evaluate_policy

def choose_environment_size():
    """
    Interfaz gráfica para elegir el tamaño del entorno.
    """
    def on_selection(map_size):
        root.destroy()  # Cierra la ventana de selección
        main_frozenlake(map_size)  # Inicia FrozenLake con el tamaño seleccionado

    # Crear la ventana principal
    root = tk.Tk()
    root.title("Seleccionar tamaño del entorno")
    root.geometry("300x150")

    # Título
    label = tk.Label(root, text="Elige el tamaño del entorno", font=("Arial", 14))
    label.pack(pady=20)

    # Botones de selección
    btn_4x4 = tk.Button(root, text="4x4", font=("Arial", 12), command=lambda: on_selection("4x4"))
    btn_4x4.pack(side=tk.LEFT, padx=20)

    btn_8x8 = tk.Button(root, text="8x8", font=("Arial", 12), command=lambda: on_selection("8x8"))
    btn_8x8.pack(side=tk.RIGHT, padx=20)

    # Mostrar la ventana
    root.mainloop()

def main_frozenlake(map_size):
    # Crear el entorno
    env = create_environment(map_size=map_size, render_mode="human")

    # Entrenar usando iteración de valores
    policy, V, deltas = value_iteration(env)

    # Guardar la política óptima y Guardar gráficas
    env_shape = (4, 4) if map_size == "4x4" else (8, 8)
    save_policy(policy)
    plot_and_save_convergence(deltas)
    plot_and_save_policy_heatmap(policy, env_shape=env_shape)
    plot_value_heatmap(V, env_shape=env_shape)

    # Evaluar política
    success_rate, average_reward, average_steps = evaluate_policy(env, policy, n_episodes=10, render=True)

    print(f"\nTasa de Éxito: {success_rate:.2f}%")
    print(f"Recompensa Promedio: {average_reward:.2f}")
    print(f"Pasos Promedio: {average_steps:.2f}")


if __name__ == "__main__":
    choose_environment_size()
