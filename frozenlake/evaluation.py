from frozenlake.utils import plot_and_save_rewards, save_metrics_to_file

def evaluate_policy(env, policy, n_episodes=100, render=False):
    """Evalúa la política y guarda las métricas y resultados."""
    total_rewards = 0
    total_steps = 0
    successful_episodes = 0
    rewards_per_episode = []
    success_count = 0

    for episode in range(1, n_episodes + 1):

        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        if render:
            print(f"\nEpisodio {episode}:")
            env.render()

        while not done:

            action = policy[state]
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if done and reward > 0:
                successful_episodes += 1

        # Verificar resultado final
        if episode_reward > 0:
            print(f"¡Meta alcanzada en {steps} pasos!")
            success_count += 1
        else:
            print(f"El agente cayó en un hueco en {steps} pasos.")

        total_rewards += episode_reward
        total_steps += steps
        rewards_per_episode.append(episode_reward)

    # Métricas
    success_rate = (successful_episodes / n_episodes) * 100
    average_reward = total_rewards / n_episodes
    average_steps = total_steps / n_episodes

    # Guardar resultados
    save_metrics_to_file(success_rate, average_steps, average_reward)
    plot_and_save_rewards(rewards_per_episode)

    return success_rate, average_reward, average_steps
