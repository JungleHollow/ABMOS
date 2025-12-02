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

    def add_agents(self, agents: list[Agent]) -> None:
        """
        Add new Agents to the Model's AgentSet.

        :param agents: A list of Agent objects to be added to the AgentSet
        """
        for agent in agents:
            self.agents.add(agent)

    def generate_agents(self, attributes: dict, number: int = 100) -> None:
        """
        Randomly generates a number of Agent objects using the given attribute dictionary.
        The dictionary items should be singular explciit values to assign for all agents, or tuples representing:
            (mean, standard deviation, distribution to use) for random generation

        :param attributes: A dictionary containing (attribute: tuple) pairs for Agent attribute setting
        :param number: Number of agents to be randomly created.
        """
        for i in range(number):
            new_agent: Agent = Agent()
            for key, value in attributes.items():
                if len(value) == 1:
                    new_agent.add_attribute(key, value=value)
                else:
                    new_agent.add_attribute(
                        key, mean=value[0], sdev=value[1], distribution=value[2]
                    )
            self.agents.add(new_agent)

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
