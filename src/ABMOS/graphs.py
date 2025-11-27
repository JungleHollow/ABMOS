from __future__ import annotations

from collections.abc import Iterable

import networkx
import numpy as np
import rustworkx

from .agents import Agent


class Graph:
    """
    A graph class that defines a single agent-based model layer.
    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.
    """

    def __init__(self):
        self.nodes: Iterable[Agent] = []
        self.edges: Iterable[Iterable[int]] = []
        self.node_count: int = 0
        self.edge_count: int = 0
        self.weights: Iterable[Iterable[float]] = []
        self.name: str = ""

    def load_graph(self, path: str) -> None:
        """
        Loads a stored Graph object from the given path.

        :param path: Path to a stored graph file.
        """
        pass

    def create_graph(
        self,
        nodes: Iterable[Agent],
        edges: Iterable[Iterable[int]],
        weights: Iterable[Iterable[float]],
        name: str | None = None,
    ):
        """
        Creates a new Graph object from the given parameters.

        :param nodes: List of nodes (agents) in the graph.
        :param edges: List of edges (relationships) between the agents in the graph.
        :param weights: List of weights (relationship strengths) for the edges.
        :param name: Optional name to label the graph as some specific social hierarchy.
        """
        pass

    def change_weights(self, node_1: Agent, node_2: Agent, value: float) -> None:
        """
        Updates the weight of the relationship between two agents in the graph.
        If no relationship previously exists, a new one is created.

        :param node_1: Some Agent in the graph.
        :param node_2: Some other Agent in the graph.
        :param value: The new weight to assign.
        """
        pass

    def remove_node(self, node: Agent) -> None:
        """
        Removes an agent from the graph, along with any relationships involving it.

        :param node: The Agent object to remove.
        """
        pass

    def add_node(
        self, node: Agent, edges: Iterable[int], weights: Iterable[float]
    ) -> None:
        """
        Adds a new agent to the graph, and creates all relationships involving it.

        :param node: The Agent object to add.
        :param edges: A list of edges between the agent and others in the graph.
        :param weights: A list of weights for the corresponding edges.
        """

        # TODO: This will accept 1D lists for edges and weights but will update the 2D matrices for the graph(s)...
        pass

    def agent_in_graph(self, agent: Agent) -> bool:
        """
        A simple function that checks wether an Agent exists within a Graph
        :param agent: the Agent whose existence in the Graph is being checked for
        """
        if agent.__in__(self.nodes):
            return True
        return False


class GraphSet:
    """
    A class that will collect all of the different social hierarchy graphs in the same structure
    and provide utilities using this collection
    """

    def __init__(self, graphs: Iterable[Graph] = []) -> None:
        self.graphs: Iterable[Graph] = graphs

    def agent_opinion_threshold(
        self, agent: Agent, threshold: float = 0.9
    ) -> Iterable[str]:
        """
        A utility function that iterates over the GraphSet and records for which social hierarchies a specific Agent's weighting
        of those hierarchies is above a certain threshold value
        :param agent: the Agent for which to check the AgentSet for
        :param threshold: the absolute threshold value over which the Agent's opinion is considered significant
        """
        significant_hierarchies: Iterable[str] = []
        for hierarchy in self.graphs:
            if hierarchy.agent_in_graph(agent):
                social_weighting: float = agent.social_weightings[hierarchy.name]
                if abs(social_weighting) > threshold:
                    significant_hierarchies.append(hierarchy.name)
        return significant_hierarchies
