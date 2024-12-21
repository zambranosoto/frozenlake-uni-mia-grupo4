# frozenlake/algorithms.py
import numpy as np

def compute_action_values(env, state, V, gamma):
    """
    Descripción:
        Calcula los valores de acción para un estado dado basado en la función de valor actual.

    Parámetros de entrada:
        env: Entorno de Gymnasium.
        state: Estado actual (índice entero).
        V: Función de valor actual (array 1D).
        gamma: Factor de descuento (float).

    Returns:
        action_values: Array 1D con los valores esperados de cada acción desde el estado dado.
    """

    action_values = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        # Iterar sobre las transiciones posibles desde el estado actual y la acción.
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            action_values[action] += prob * (reward + gamma * V[next_state])
    return action_values

def update_value_function(env, V, gamma, theta):
    """
    Descripción:
        Itera sobre la función de valor hasta la convergencia usando Iteración de Valores.

    Parámetros de entrada:
        env: Entorno de Gymnasium.
        V: Función de valor inicial (array 1D).
        gamma: Factor de descuento (float).
        theta: Umbral para determinar la convergencia (float).

    Returns:
        V: Función de valor óptima (array 1D).
    """

    while True:
        delta = 0  # Cambio máximo entre iteraciones
        for state in range(env.observation_space.n):
            # Calcula los valores de acción para el estado actual
            action_values = compute_action_values(env, state, V, gamma)
            max_value = max(action_values)
            # Calcula el cambio absoluto máximo
            delta = max(delta, abs(max_value - V[state]))
            # Actualiza la función de valor para el estado actual
            V[state] = max_value
        # Verifica si el cambio máximo es menor que el umbral para detener la iteración
        if delta < theta:
            break
    return V

def extract_policy(env, V, gamma=0.99):
    """
    Descripción:
        Extrae la política óptima a partir de la función de valor.

    Parámetros de entrada:
        env: Entorno de Gymnasium.
        V: Función de valor óptima (array 1D).
        gamma: Factor de descuento (float).

    Returns:
        policy: Política óptima (array 1D con índices de las acciones óptimas para cada estado).
    """

    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        # Calcula los valores de acción para cada estado
        action_values = compute_action_values(env, s, V, gamma)
        # Selecciona la acción con el valor más alto
        policy[s] = np.argmax(action_values)

    return policy

def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Descripción:
        Implementa la iteración de valores para encontrar la política óptima.

    Parámetros de entrada:
        env: Entorno de Gymnasium.
        gamma: Factor de descuento (float, por defecto 0.99).
        theta: Umbral para determinar la convergencia (float, por defecto 1e-6).

    Returns:
        policy: Política óptima (array 1D).
        V: Función de valor óptima (array 1D).
        deltas: Lista de cambios máximos por iteración (para análisis de convergencia).
    """

    n_states = env.observation_space.n
    V = np.zeros(n_states)  # Inicializa la función de valor con ceros
    deltas = []  # Lista para almacenar el cambio máximo por iteración

    while True:
        delta = 0  # Cambio máximo para la iteración actual
        new_V = np.copy(V)  # Copia de la función de valor actual
        for s in range(n_states):
            # Calcula los valores de acción para el estado actual
            action_values = compute_action_values(env, s, V, gamma)
            # Actualiza el valor del estado con el máximo de los valores de acción
            new_V[s] = max(action_values)
            # Calcula el cambio absoluto máximo
            delta = max(delta, abs(new_V[s] - V[s]))
        deltas.append(delta)  # Almacena el cambio máximo de la iteración actual
        V = new_V  # Actualiza la función de valor
        # Verifica si el cambio máximo es menor que el umbral para detener la iteración
        if delta < theta:
            break

    # Extrae la política óptima a partir de la función de valor convergida
    policy = extract_policy(env, V, gamma)
    return policy, V, deltas

