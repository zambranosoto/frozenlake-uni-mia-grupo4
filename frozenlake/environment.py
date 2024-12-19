# frozenlake/environment.py
import gymnasium as gym


def create_environment(render_mode=None):
    """
    Crea y retorna el entorno FrozenLake-v1.

    Args:
        render_mode (str): Modo de renderización ('human', 'rgb_array', None).

    Returns:
        env: Instancia del entorno Gymnasium.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=render_mode)
    return env