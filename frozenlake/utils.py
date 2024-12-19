# frozenlake/utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_policy(policy, filename="policy.npy"):
    np.save(filename, policy)
    print(f"Política guardada en {filename}")

def load_policy(filename="policy.npy"):
    return np.load(filename)

def plot_convergence(value_history):
    plt.plot(value_history)
    plt.xlabel("Iteraciones")
    plt.ylabel("Cambio máximo en V")
    plt.title("Convergencia del valor óptimo")
    plt.grid()
    plt.show()

def create_results_folder():
    """Crea la carpeta results si no existe."""
    if not os.path.exists("results"):
        os.makedirs("results")

def plot_and_save_convergence(delta_values, filename="convergence.png"):
    """Genera y guarda la gráfica de convergencia."""
    create_results_folder()
    plt.figure(figsize=(8, 6))
    plt.plot(delta_values, label="Delta (Máx)")
    plt.xlabel("Iteraciones")
    plt.ylabel("Delta (Máx)")
    plt.title("Convergencia de Iteración de Valores")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{filename}")
    plt.close()

def plot_and_save_policy_heatmap(policy, env_shape=(4, 4), filename="policy_heatmap.png"):
    """Genera y guarda un mapa de calor para la política óptima."""
    create_results_folder()
    policy_grid = np.reshape(policy, env_shape)
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy_grid, annot=True, cbar=False, cmap="coolwarm", fmt="d")
    plt.title("Mapa de Calor de la Política Óptima")
    plt.savefig(f"results/{filename}")
    plt.close()

def plot_and_save_rewards(average_rewards, filename="average_rewards.png"):
    """Genera y guarda la gráfica de recompensa promedio."""
    create_results_folder()
    plt.figure(figsize=(8, 6))
    plt.plot(average_rewards, label="Recompensa Promedio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Promedio")
    plt.title("Recompensa Promedio por Episodio")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{filename}")
    plt.close()

def save_metrics_to_file(success_rate, average_steps, average_reward, filename="metrics.txt"):
    """Guarda las métricas de evaluación en un archivo de texto."""
    create_results_folder()
    with open(f"results/{filename}", "w") as f:
        f.write("Métricas de Evaluación:\n")
        f.write(f"Tasa de Éxito: {success_rate:.2f}%\n")
        f.write(f"Pasos Promedio: {average_steps:.2f}\n")
        f.write(f"Recompensa Promedio: {average_reward:.2f}\n")

def plot_value_heatmap(V, env_shape=(4, 4), filename="value_heatmap.png"):
    V_matrix = np.reshape(V, env_shape)
    sns.heatmap(V_matrix, annot=True, cmap="YlGnBu", cbar=True)
    plt.title("Mapa de calor de los valores de los estados")
    plt.savefig(f"results/{filename}")
    plt.close()
