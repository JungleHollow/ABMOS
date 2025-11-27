from __future__ import annotations

from collections.abc import Iterable

from .agents import Agent
from .graphs import Graph


class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.
    """

    def __init__(self):
        self.graphs: Iterable[Graph] = []
        self.agents: Iterable[Agent] = []
        self.iteration: int = 0

    def init_graphs(self, graphs: Iterable[Graph | str]) -> None:
        """
        Defines the graphs to be used for the model.

        :param graphs: A list of either created Graph objects, or relevant filenames.
        """
        pass

    def init_agents(
        self, number: int = 100, agents: Iterable[Agent] | None = None
    ) -> None:
        """
        Defines the agents to be used for the model.

        :param number: Number of agents to be randomly created.
        :param agents: List of created Agent objects.
        """
        pass

    def step(self) -> None:
        """
        Steps the model forward one iteration.
        """
        pass

    def update(self) -> None:
        """
        Updates the agents' internal states to match the model step.
        """
        pass

    def visualise(self) -> None:
        """
        Visualises the agents' internal states.
        """
        pass
