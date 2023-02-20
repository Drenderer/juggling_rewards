"""
    mail@kaiploeger.net
"""


import numpy as np


def survival_bonus():
    """Reward for staying alive. Aka not dropping the ball.
    """
    return 1.0


def control_penalty(tau):
    """Reward for using less control effort.
    """
    return - np.linalg.norm(tau)**2


def ball_distance_penalty(pos_ball0, pos_ball1, diameter=0.075):
    """Reward for keeping the balls separated. Important to encourange throwing.
    """
    dist = np.linalg.norm(pos_ball0 - pos_ball1)
    return 1 / (diameter - dist)

