from __future__ import annotations

from random import Random

import numpy as np

from .agents import Agent
from .model import ABModel


class AgentSpace:
    """
    A class that is used to represent the environment in which the Model Agents are interacting with each other.
    """

    def __init__(
        self,
        model: ABModel,
        xlims: tuple[int, int] = (0, 100),
        ylims: tuple[int, int] = (0, 100),
        allow_agent_overlap: bool = True,
        max_agents_per_grid: int = 4,
        agent_sphere_of_influence: float = 1.0,
        space_type: str = "discrete",
    ) -> None:
        """
        Initialise the AgentSpace for the Model. By default, a 100x100 2-dimensional space is used.

        :param model: The parent ABModel object that the AgentSpace object is being attached to
        :param xlims: The lower and upper bounds of the space's x-axis
        :param ylims: The lower and upper bounds of the space's y-axis
        :param allow_agent_overlap: Flag whether more than one Agent can be in the same grid simultaneously
        :param max_agents_per_grid: The maximum number of Agents that can be in the same grid simultaneously
        :param agent_sphere_of_influence: The maximum distance between agents at which inter-Agent interaction can occur
        :param space_type: The type of space that is used for the Model representation. Current supported types are:
            - "discrete" -- A 2-dimensional (n x m) grid space where n is the size of the x-axis and m is the size of the y-axis
            - "continuous" -- A 2-dimensional gridless space in which Agents can be located at any arbitrary point within the axis limits
        """
        self.parent_model: ABModel = model
        self.xlims: tuple[int, int] = xlims
        self.ylims: tuple[int, int] = ylims
        self.allow_agent_overlap: bool = allow_agent_overlap
        self.max_agents_per_grid: int = max_agents_per_grid
        self.agent_sphere_of_influence: float = agent_sphere_of_influence
        self.space_type: str = space_type
        self.space: np.ndarray = np.zeros(
            (
                self.xlims[1] - self.xlims[0],
                self.ylims[1] - self.ylims[0],
                self.max_agents_per_grid,
            ),
            dtype=int,
        )

    def get_limits(self) -> dict[str, tuple[float, float]]:
        """
        A getter method that returns the AgentSpace map limits as an organised dictionary.
        """
        return {"xlims": self.xlims, "ylims": self.ylims}

    def check_neighbours(self, agent: Agent) -> list[int]:
        """
        A function that checks the 8 cells directly adjacent to an agent, and returns a list
        of length 8 with the number of neighbours in each cell; left to right, top to bottom.

        :param agent: The Agent for which the neighbours are being checked
        """
        num_neighbours = [0 for _ in range(8)]
        agent_position: tuple[int, int] = agent.position
        cell_counter: int = 0
        for i in range(agent_position[0] - 1, agent_position[0] + 2):
            for j in range(agent_position[1] - 1, agent_position[1] + 2):
                if i == agent_position[0] and j == agent_position[1]:
                    continue
                else:
                    num_neighbours[cell_counter] = np.sum(self.space[i, j])
        return num_neighbours

    def move_agent(self, agent: Agent) -> tuple[int, int]:
        """
        A function that determines what a valid movement would be for a given Agent,
        and returns a tuple with the proposed new coordinates

        :param agent: The Agent that is moving
        """
        previous_position: tuple[int, int] = agent.position
        num_neighbours = self.check_neighbours(agent)
        possible_moves: list[int] = []
        for i in range(len(num_neighbours)):
            if num_neighbours[i] < self.max_agents_per_grid:
                possible_moves.append(i)
        if len(num_neighbours) <= 0:
            return previous_position
        else:
            chosen_move = Random.choice(Random(), possible_moves)
            move_tuple: tuple[int, int] = (0, 0)
            match chosen_move:
                case 0:
                    move_tuple = (-1, -1)
                case 1:
                    move_tuple = (0, -1)
                case 2:
                    move_tuple = (1, -1)
                case 3:
                    move_tuple = (-1, 0)
                case 4:
                    move_tuple = (1, 0)
                case 5:
                    move_tuple = (-1, 1)
                case 6:
                    move_tuple = (0, 1)
                case 7:
                    move_tuple = (1, 1)
            return (
                previous_position[0] + move_tuple[0],
                previous_position[1] + move_tuple[1],
            )

    def __str__(self) -> str:
        """
        An override to what calling `print()` on this object will output
        """
        return f"AgentSpace of {self.space_type} type with size of {self.xlims[1]} by {self.ylims[1]} units and permitting Agent interaction at distances of up to {self.agent_sphere_of_influence} units"
