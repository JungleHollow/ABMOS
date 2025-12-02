from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .agents import Agent, AgentSet
from .graphs import Graph, GraphSet
from .logging import ABMOSLogger


class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.
    """

    def __init__(self, iterations: int = 100):
        self.graphs: GraphSet = GraphSet()
        self.agents: AgentSet = AgentSet()
        self.logger: ABMOSLogger = ABMOSLogger()
        self.current_iteration: int = 0
        self.max_iterations: int = iterations

    def add_graphs(self, graphs: list[Any], names: list[str]) -> None:
        """
        Add new Graphs to the Model's GraphSet.

        :param graphs: A list of Graph objects or filepaths to stored GraphML objects
        :param names: A list of the corresponding social hierarchy names to give to the Graphs
        """
        if type(graphs[0]) is Graph:
            for graph in graphs:
                self.graphs.add_graph(graph)
        else:
            for idx, graph in enumerate(graphs):
                new_graph: Graph = Graph(names[idx])
                new_graph.load_graph(graph, names[idx])
                self.graphs.add_graph(new_graph)

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
