from frozenlake.utils import plot_and_save_rewards, save_metrics_to_file

def evaluate_policy(env, policy, n_episodes=100, render=False):
    """
    Descripción:
        Evalúa una política en el entorno FrozenLake-v1 y guarda las métricas de rendimiento.

    Parámetros de entrada:
        env: Entorno de Gymnasium donde se evaluará la política.
        policy: Política a evaluar (array 1D con índices de acciones por estado).
        n_episodes: Número de episodios para evaluar la política (int, por defecto 100).
        render: Si es True, muestra el entorno en cada paso del episodio (bool, por defecto False).

    Returns:
        success_rate: Tasa de éxito (float, porcentaje de episodios en los que el agente alcanzó la meta).
        average_reward: Recompensa promedio obtenida por episodio (float).
        average_steps: Número promedio de pasos por episodio (float).
    """
    total_rewards = 0  # Suma total de recompensas obtenidas en todos los episodios
    total_steps = 0  # Suma total de pasos en todos los episodios
    successful_episodes = 0  # Número de episodios exitosos
    rewards_per_episode = []  # Lista para almacenar las recompensas por episodio
    success_count = 0  # Cuenta de episodios exitosos (recompensa > 0)

    for episode in range(1, n_episodes + 1):
        # Reinicia el entorno y obtiene el estado inicial
        state, _ = env.reset()
        done = False
        episode_reward = 0  # Recompensa acumulada del episodio
        steps = 0  # Número de pasos en el episodio

        if render:
            print(f"\nEpisodio {episode}:")
            env.render()

        while not done:
            # Selecciona la acción basada en la política
            action = policy[state]
            # Ejecuta la acción y obtiene la transición resultante
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # Incrementa el contador de episodios exitosos si se alcanza la meta
            if done and reward > 0:
                successful_episodes += 1

        # Mensaje de estado al finalizar el episodio
        if episode_reward > 0:
            print(f"¡Meta alcanzada en {steps} pasos!")
            success_count += 1
        else:
            print(f"El agente cayó en un hueco en {steps} pasos.")

        total_rewards += episode_reward
        total_steps += steps
        rewards_per_episode.append(episode_reward)

    # Cálculo de métricas
    success_rate = (successful_episodes / n_episodes) * 100  # Tasa de éxito en porcentaje
    average_reward = total_rewards / n_episodes  # Recompensa promedio por episodio
    average_steps = total_steps / n_episodes  # Número promedio de pasos por episodio

    # Guardar métricas y resultados
    save_metrics_to_file(success_rate, average_steps, average_reward)
    plot_and_save_rewards(rewards_per_episode)

    return success_rate, average_reward, average_steps
