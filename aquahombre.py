from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import enum

_SCREEN = "screen"
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_STOP = actions.FUNCTIONS.Stop_quick.id
_NO_OP = actions.FUNCTIONS.no_op.id

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_SELECT_ALL = [0]
_NOT_QUEUED = [0]

class AbstractAction(enum.IntEnum):
    MOVE_UP           = 0
    MOVE_UP_RIGHT     = 1
    MOVE_RIGHT        = 2
    MOVE_DOWN_RIGHT   = 3
    MOVE_DOWN         = 4
    MOVE_DOWN_LEFT    = 5
    MOVE_LEFT         = 6
    MOVE_UP_LEFT      = 7
    ATTACK_UP         = 8
    ATTACK_UP_RIGHT   = 9
    ATTACK_RIGHT      = 10
    ATTACK_DOWN_RIGHT = 11
    ATTACK_DOWN       = 12
    ATTACK_DOWN_LEFT  = 13
    ATTACK_LEFT       = 14
    ATTACK_UP_LEFT    = 15
    STOP              = 16

class Direction(enum.IntEnum):
    UP         = 0
    UP_RIGHT   = 1
    RIGHT      = 2
    DOWN_RIGHT = 3
    DOWN       = 4
    DOWN_LEFT  = 5
    LEFT       = 6
    UP_LEFT    = 7

class AquaHombre(base_agent.BaseAgent):

    def step(self, obs):
        super(AquaHombre, self).step(obs)

        abstract_action = Policy.sample_actions(obs) # TODO: implement Perhaps sample from action distribution

        if not self.is_action_possible(abstract_action, obs):
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        action_realizer = ActionRealizer(obs)
        action, params = action_realizer.realize_action(abstract_action)
        return actions.FunctionCall(action, params)

    def is_action_possible(self, abstract_action, obs):
        available_actions = obs.observation["available_actions"]
        if abstract_action < AbstractAction.ATTACK_UP and _MOVE_SCREEN in available_actions:
            return True
        elif abstract_action < AbstractAction.STOP and _STOP in available_actions:
            return True
        elif abstract_action == AbstractAction.STOP and _STOP in available_actions:
            return True
        else:
            return False


class Policy():

    @staticmethod
    def sample_actions(obs):
        return random.choice([AbstractAction.MOVE_UP_RIGHT])

class ActionRealizer():

    def __init__(self, obs):
        self.obs = obs

    def realize_action(self, action_id):
        if not action_id and action_id not in [action.value for action in AbstractAction]:
            raise ValueError("action id was None or it wasn't a valid action")

        if action_id == AbstractAction.STOP: # if it's movement action
            return (_STOP, [_NOT_QUEUED]) # TODO: check
        else:
            return self.handle_movement(action_id)

    def handle_movement(self, action_id):
        number_of_directions = 8
        direction   = action_id if action_id < AbstractAction.ATTACK_UP else action_id - number_of_directions
        action_type = _MOVE_SCREEN if action_id < AbstractAction.ATTACK_UP else _ATTACK_SCREEN

        target_position = self.get_target_position(direction)
        return (action_type, [_NOT_QUEUED, target_position]) # (action, [position])

    def get_target_position(self, direction):
        marine_positions = self.find_marine_positions()

        target_ys, target_xs = np.copy(marine_positions)
        target_ys, target_xs  = int(target_ys.mean()), int(target_xs.mean()) # To compensate for marine occupying  multiple tiles

        if direction in [Direction.UP, Direction.UP_LEFT, Direction.UP_RIGHT]:
            target_ys = max(0, target_ys - 5)
        if direction in [Direction.DOWN, Direction.DOWN_LEFT, Direction.DOWN_RIGHT]:
            target_ys = min(83, target_ys + 5) #TODO: find maximum value via scree features
        if direction in [Direction.LEFT, Direction.DOWN_LEFT, Direction.UP_LEFT]:
            target_xs = max(0, target_xs - 5)
        if direction in [Direction.RIGHT, Direction.DOWN_RIGHT, Direction.UP_RIGHT]:
            target_xs = min(83, target_xs + 5)
        return target_xs, target_ys # Remember that even though observation gives y,x coordinates. We give back x,y!

    def find_marine_positions(self):
        """ Returns tuple: (np.array[y_positions], np.array[x_positions]) """ 
        player_relative = self.obs.observation[_SCREEN][_PLAYER_RELATIVE]
        return (player_relative == _PLAYER_FRIENDLY).nonzero()
