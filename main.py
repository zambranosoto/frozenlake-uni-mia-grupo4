# main.py
from frozenlake.environment import create_environment
from frozenlake.algorithms import value_iteration
from frozenlake.utils import save_policy, plot_and_save_convergence, plot_and_save_policy_heatmap, plot_value_heatmap
from frozenlake.evaluation import evaluate_policy


def main_frozenlake():
    # Crear el entorno
    env = create_environment(render_mode="human")

    # Entrenar usando iteración de valores
    policy, V, deltas = value_iteration(env)

    # Guardar la política óptima
    save_policy(policy)
    plot_and_save_convergence(deltas)
    plot_and_save_policy_heatmap(policy)
    plot_value_heatmap(V)

    # Visualizar el mapa de calor de los valores
    # plot_value_heatmap(V)

    # Evaluar política
    success_rate, average_reward, average_steps = evaluate_policy(env, policy, n_episodes=10, render=True)

    print(f"\nTasa de Éxito: {success_rate:.2f}%")
    print(f"Recompensa Promedio: {average_reward:.2f}")
    print(f"Pasos Promedio: {average_steps:.2f}")


if __name__ == "__main__":
    main_frozenlake()
