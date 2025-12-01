from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx

from .agents import Agent


class GraphNode:
    """
    A helper class that allows rustworx to more efficiently store information about Agents in the graph nodes
    """

    def __init__(self, agent: Agent):
        self.index: int | None = None
        self.agent: Agent = agent

    def __str__(self):
        return f"Agent ({self.agent.id}) at graph node ({self.index})"


class GraphEdge:
    """
    A helper class that allows rustworx to more efficiently store information about Agent relationships in the graph edges
    """

    def __init__(
        self,
        weighting: float | None = None,
        from_node: int | None = None,
        to_node: int | None = None,
        hierarchy: str | None = None,
    ):
        self.index: int | None = None
        self.weighting: float | None = weighting
        self.from_node: int | None = from_node
        self.to_node: int | None = to_node
        self.hierarchy: str | None = hierarchy

    def __str__(self):
        return f"GraphEdge of weight ({self.weighting}) from node ({self.from_node}) to node ({self.to_node}) in the {self.hierarchy} social layer"


class Graph:
    """
    A graph class that defines a single agent-based model layer.
    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.
    """

    def __init__(self):
        # TODO: Replace this with networkx and rustworx
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
        return agent.__in__(self.nodes)


class GraphSet:
    """
    A class that will collect all of the different social hierarchy graphs in the same structure
    and provide utilities using this collection
    """

    def __init__(self, graphs: Iterable[Graph] = []) -> None:
        self.graphs: pl.Series = pl.Series(graphs)

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
