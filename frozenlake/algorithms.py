# frozenlake/algorithms.py
import numpy as np

def compute_action_values(env, state, V, gamma):
    action_values = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            action_values[action] += prob * (reward + gamma * V[next_state])
    return action_values

def update_value_function(env, V, gamma, theta):
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            action_values = compute_action_values(env, state, V, gamma)
            max_value = max(action_values)
            delta = max(delta, abs(max_value - V[state]))
            V[state] = max_value
        if delta < theta:
            break
    return V

def extract_policy(env, V, gamma=0.99):
    """
    Extrae la política óptima a partir de la función de valor.
    Args:
        env: Entorno de Gymnasium.
        V: Función de valor óptima.
        gamma: Factor de descuento.
    Returns:
        policy: Política óptima.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = compute_action_values(env, s, V, gamma)
        policy[s] = np.argmax(action_values)

    return policy

def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Implementa la iteración de valores.
    Args:
        env: Entorno de Gymnasium.
        gamma: Factor de descuento.
        theta: Umbral para determinar la convergencia.
    Returns:
        policy: Política óptima.
        V: Función de valor óptima.
        deltas: Lista con los cambios máximos por iteración.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    deltas = []  # Lista para guardar los cambios máximos

    while True:
        delta = 0
        new_V = np.copy(V)
        for s in range(n_states):
            action_values = compute_action_values(env, s, V, gamma)
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break

    policy = extract_policy(env, V, gamma)
    return policy, V, deltas

