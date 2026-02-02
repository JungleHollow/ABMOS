from __future__ import annotations

from typing import Any

import polars as pl

from src.ABMOS.agents import Agent, AgentSet
from src.ABMOS.graphs import Graph, GraphEdge, GraphNode, GraphSet
from src.ABMOS.model import ABModel


class DataReader:
    """
    A class that will be used to create and handle an appropriate ABMOS model from the given agent and
    social hierarchy CSV file paths
    """

    def __init__(
        self,
        agents_path: str,
        initial_hierarchies: list[str],
        social_path: str,
        opinions_path: str | None = None,
        iterations: int = 100,
        xlims: tuple[int, int] | None = None,
        ylims: tuple[int, int] | None = None,
    ):
        """
        :param agents_path: A relative or absolute file path to a CSV file containing relevant data on model agent characteristics
        :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the reader
        :param social_path: A relative or absolute file path to a CSV file containing relevant data on the relative influence of the existing social hierarchies in the community
        :optional param opinions_path: A relative or absolute file path to a CSV file containing the dependant variables of actual agent opinions; used to compare model accuracy after execution
        :optional param iterations: The number of iterations that the ABModel will be run for
        :optional param xlims: The x-axis boundaries of the agent space within the ABModel
        :optional param ylims: The y-axis boundaries of the agent space within the ABModel
        """
        self.agents_path: str = agents_path
        self.agents_df: pl.DataFrame
        self.initial_hierarchies: list[str] = initial_hierarchies
        self.hierarchy_influences: dict[str, dict] = {}
        self.social_path: str = social_path
        self.social_df: pl.DataFrame
        self.opinions_path: str | None = opinions_path
        self.opinions_df: pl.DataFrame

        with open(self.agents_path, "r") as file:
            self.agents_df = pl.read_csv(file)
        with open(self.social_path, "r") as file:
            self.social_df = pl.read_csv(file)

        if self.opinions_path:
            with open(self.opinions_path, "r") as file:
                self.opinions_df = pl.read_csv(file)

        self.model: ABModel = ABModel(iterations=iterations, xlims=xlims, ylims=ylims)

    def extract_hierarchy_influences(self):
        """
        Calculates the influence of each social hierarchy for each agent
        """
        general_column: pl.Series | None = self.social_df.get_column(
            "General", default=None
        )
        if general_column is not None:
            for hierarchy in set(
                list(general_column)
            ):  # set() to reduce the number that will be iterated over
                if hierarchy not in self.initial_hierarchies:
                    self.initial_hierarchies.append(hierarchy)

        for agent_row in self.social_df.iter_rows(named=True):
            hierarchy_effects: dict = {}
            for hierarchy in self.initial_hierarchies:
                hierarchy_effects[hierarchy] = (
                    0.0  # Initialise each hierarchy effect, even if not explicitly seen in the current agent row
                )

            raw_hierarchy_values: dict = {}
            for key, value in agent_row:
                if key == "AgenteId":
                    continue
                elif key == "General":
                    hierarchy_effects[value] = 1.0  # Placeholder of 1.0 strength for the moment
                else:
                    hierarchy_name: str = key.split("_")[0]
                    if hierarchy_name not in raw_hierarchy_values.keys():
                        raw_hierarchy_values[hierarchy_name] = []
                    raw_hierarchy_values[hierarchy_name].append(abs(int(value)))
            
            for key, value in raw_hierarchy_values:
                sum_values: int = sum(value)
                averaged_sum: float = sum_values / len(value)
                final_effect: float = averaged_sum / 10.0
                hierarchy_effects[key] = final_effect
                    
            self.hierarchy_influences[agent_row["AgenteId"]] = hierarchy_effects

    def create_model_agents(self):
        """
        Uses agents_df and the extracted hierarchy influences to create Agent objects for the ABModel
        """
        pass

    def create_model_graphs(self):
        """
        Uses initial_hierarchies and the extracted hierarchy influences to create Graph objects with the appropriate GraphNodes
        """
        pass
