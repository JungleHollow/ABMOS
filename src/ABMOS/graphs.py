from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx
from _typeshed import Incomplete as Incomplete
from rustworkx.rustworkx import NodeIndices

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
        hierarchy: str,
        from_node: int,
        to_node: int,
        weighting: float = 0.0,
    ):
        self.index: int
        self.weighting: float = weighting
        self.from_node: int = from_node
        self.to_node: int = to_node
        self.hierarchy: str = hierarchy

    def __str__(self):
        return f"GraphEdge of weight ({self.weighting}) from node ({self.from_node}) to node ({self.to_node}) in the {self.hierarchy} social layer"


class Graph:
    """
    A graph class that defines a single agent-based model layer.
    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.
    """

    def __init__(self, name: str):
        # Defined as DiGraph as it is common in social networks for relationships to be unidirectional or unbalanced
        self.graph: rx.PyDiGraph = rx.PyDiGraph()
        self.node_count: int = 0
        self.edge_count: int = 0
        self.name: str = name

    def load_graph(self, path: str, name: str) -> None:
        """
        Loads a Graph object stored in the GraphML format from the given path.
        The social hierarchy name must be explicitly passed with this call.

        :param path: Path to a stored graph file.
        """
        graph: list[Any] = rx.read_graphml(path)
        self.graph = graph[0]
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())
        self.name = name

    def save_graph(self, path: str) -> None:
        """
        Saves the existing Graph object in the GraphML format to the given path.

        :param path: Path to which the Graph will be saved.
        """
        rx.write_graphml(self.graph, path)

    def get_node(self, node_index: int) -> GraphNode:
        """
        A getter function to access GraphNode objects

        :param node_index: The index of the node to access
        """
        return self.graph.nodes()[node_index]

    def get_edge(self, edge_index: int) -> GraphEdge:
        """
        A getter function to access GraphEdge objects

        :param edge_index: The index of the edge to access
        """
        return self.graph.edges()[edge_index]

    def update_node_indices(self):
        """
        Iterates over all the existing nodes in the graph and updates their stored indices to reflect the current graph state.
        Will also update the graph node_count attribute
        """
        for index in self.graph.node_indices():
            self.graph[index].index = index
        self.node_count = len(self.graph.nodes())

    def add_nodes(self, agents: Iterable[Agent]):
        """
        Creates appropriate GraphNodes from the given Agents, and then adds these to the graph.

        :param agents: The Agent objects that will be converted to GraphNodes and added to the graph
        """
        nodes = []
        for agent in agents:
            agent_node = GraphNode(agent)
            nodes.append(agent_node)

        self.graph.add_nodes_from(nodes)
        self.update_node_indices()

    def update_edge_indices(self):
        """
        Iterates over all the existing edges in the graph and updates their stored indices to reflect the current graph state.
        Will also update the graph edge_count attribute
        """
        for index, data in self.graph.edge_index_map().items():
            data[2].index = index
        self.edge_count = len(self.graph.edges())

    def add_edges(self, edges: dict):
        """
        Creates appropriate GraphEdges from the given dictionary and then adds these to the graph.

        :param edges: A dictionary of key-list pairs where each key corresponds to (from_node, to_node, [optional] weighting)
        """
        graph_edges = []
        from_nodes = edges["from_node"]
        to_nodes = edges["to_node"]
        weightings = None
        if "weighting" in edges.keys():
            weightings = edges["weighting"]

        if weightings:
            for i in range(len(from_nodes)):
                edge = GraphEdge(self.name, from_nodes[i], to_nodes[i], weightings[i])
                graph_edges.append(edge)
        else:
            for i in range(len(from_nodes)):
                edge = GraphEdge(self.name, from_nodes[i], to_nodes[i])
                graph_edges.append(edge)

        self.graph.add_edges_from(graph_edges)
        self.update_edge_indices()

    def relationship_exists(self, node_1: int, node_2: int) -> int | None:
        """
        Checks for the existence of a relationship (weighted edge) between two Agents (nodes)

        :param node_1: the node index of Agent 1
        :param node_2: the node index of Agent 2
        """
        for edge in self.graph.edges():
            if (edge.from_node == node_1 and edge.to_node == node_2) or (
                edge.from_node == node_2 and edge.to_node == node_1
            ):
                return edge.index
        return None

    def get_relationships(
        self, node_1: int, node_2: int
    ) -> dict[tuple[int, int], float] | None:
        """
        Return a dictionary with the bidirectional edge weightings between two nodes if they exist

        :param node_1: the node index of Agent 1
        :param node_2: the node index of Agent 2
        """
        if not self.relationship_exists(node_1, node_2):
            return None

        relationships_dict: dict[tuple[int, int], float] = {}

        with contextlib.suppress(KeyError):
            relationships_dict[(node_2, node_1)] = self.graph.adj_direction(
                node_1, True
            )[node_2]

        with contextlib.suppress(KeyError):
            relationships_dict[(node_1, node_2)] = self.graph.adj_direction(
                node_1, False
            )[node_2]

        return relationships_dict

    def change_weights(self, node_1: int, node_2: int, value: float) -> None:
        """
        Updates the weight of the relationship between two agents in the graph.
        If no relationship previously exists, a new one is created.

        :param node_1: Some Agent in the graph.
        :param node_2: Some other Agent in the graph.
        :param value: The new weight to assign.
        """
        edge_index: int | None = self.relationship_exists(node_1, node_2)
        updated_edge: list[Any] = [GraphEdge(self.name, node_1, node_2, value)]
        if edge_index is not None:
            self.graph.update_edge_by_index(edge_index, updated_edge)
        else:
            self.graph.add_edges_from(updated_edge)
        self.update_edge_indices()

    def remove_node(self, node: int) -> None:
        """
        Removes a node from the graph, along with any relationships involving it.

        :param node: The node index to remove from the graph.
        """
        self.graph.remove_node(node)

        edges_to_remove = []
        for edge in self.graph.edges():
            if edge.from_node == node or edge.to_node == node:
                edges_to_remove.append((edge.from_node, edge.to_node))

        for edge in edges_to_remove:
            self.remove_edge(edge[0], edge[1])
        # No need to update indices, as rustworkx will automatically add new nodes/edges into the largest empty index

    def remove_edge(self, node_1: int, node_2: int) -> None:
        """
        Removes a single edge from the graph.

        :param node_1: the from_node in the edge
        :param node_2: the to_node in the edge
        """
        self.graph.remove_edge(node_1, node_2)

    def agent_in_graph(self, agent: Agent) -> bool:
        """
        A simple function that checks wether an Agent exists within a Graph
        :param agent: the Agent whose existence in the Graph is being checked for
        """
        for node in self.graph.nodes():
            if agent == node.agent:
                return True
        return False


class GraphSet:
    """
    A class that will collect all of the different social hierarchy graphs in the same structure
    and provide utilities using this collection
    """

    def __init__(self, graphs: Iterable[Graph] = []) -> None:
        self.graphs: pl.Series = pl.Series(graphs)

    def list_hierarchies(self, print_out: bool = False) -> list[str]:
        """
        A utility function that iterates over the GraphSet and prints out the names of all the social hierarchies that are present
        """
        social_hierarchies: list[str] = []
        for graph in self.graphs:
            social_hierarchies.append(graph.name)

        if print_out:
            print(
                f"\nSocial hierarchies present in the GraphSet:\n\t{social_hierarchies}\n\n"
            )

        return social_hierarchies

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
