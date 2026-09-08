from __future__ import annotations

import gc
import os
import random as rd
from copy import deepcopy
from re import L
from typing import TypedDict

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class RandomWalkTester:
    """
    A test class that sets up models which use a variety of random walk distributions for their dynamic
    relationships and hierarchy weighting mechanisms.

    The main scenarios are:
        - A 'base' model with identical random walk parameters for all dynamic relationships and hierarchy weightings
        - 'HIER' -- identical parameters for dynamic relationships and unique parameters for hierarchy weightings
        - 'RELS' -- unique parameters for dynamic relationships and identical parameters for all hierarchy weightings
        - 'BOTH' -- unique parameters for both dynamic relationships and hierarchy weightings

    For each one of these models, there will also be an exploration of the effects that different strengths of parameters
    may have on emergent behaviour; i.e. positive vs. negative distribution means, small vs. large variances,
    parameter strength tied to agent influentiality, etc.

    Each instance will use identical model parameters, graphs, agent, and group populations; with only the random walk
    parameters being different between them.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading existing experiment results.
        :type existing: bool, optional
        """
        self.model_names: list[str] = TEST_PARAMETERS["model_names"]
        self.n_agents: int = AGENT_PARAMETERS["n_agents"]
        self.n_groups: int = GROUP_PARAMETERS["n_groups"]

        self.existing: bool = existing

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        for model_name in self.model_names:
            new_model: md.ABModel = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"],
                # The rw params passed to each model are overriden by the explicit ones set for agents...
                list(TEST_PARAMETERS["shared_hierarchy_rw"].values()),
                save_dir=SAVEDIRS[model_name],
                data_file=DATAFILES[model_name],
                model_id=model_name,
                simulate_groups=True,
            )
            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
            _ = gc.collect()

        # Define the dicts that will contain the populations of Agents, Graphs, and Groups
        self.model_agents: dict[str, list[agt.Agent]] = {}
        self.model_graphs: dict[str, list[gr.Graph]] = {}
        self.model_groups: dict[str, list[grp.Group]] = {}

        if not self.existing:
            self.model_agents["BASE"] = self.create_agents()
            # No changes in the Agents from the base case
            self.model_agents["RELS"] = deepcopy(self.model_agents["BASE"])

            # HIER model requires all agents to have unique hierarchy rw params
            self.model_agents["HIER"] = deepcopy(self.model_agents["BASE"])
            for agent in self.model_agents["HIER"]:
                agent.rw_distributions = TEST_PARAMETERS["unique_hierarchy_rw"]

            # Agents require unique hierarchy params
            self.model_agents["BOTH"] = deepcopy(self.model_agents["HIER"])

            print("==== All model Agents successfully created ====")

            # Create the BASE graphs
            self.model_graphs["BASE"] = self.create_graphs(self.model_agents["BASE"])

            self.model_graphs["RELS"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["RELS"]:
                # First, update all the nodes with the appropriate Agent objects
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["RELS"][idx])
                # For this case, unique rw parameters must be generated for each edge
                for edge in graph.graph.edges():
                    generated_mean: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][0][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][0][1],
                    )
                    generated_variance: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][1][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][1][1],
                    )
                    edge.set_rw_params((generated_mean, generated_variance))

            self.model_graphs["HIER"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["HIER"]:
                # Only the nodes must be updated for this case
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["HIER"][idx])

            self.model_graphs["BOTH"] = deepcopy(self.model_graphs["RELS"])
            for graph in self.model_graphs["BOTH"]:
                # Only the nodes must be updated as unique rels were already generated for RELS
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["BOTH"][idx])

            print("==== All model Graphs successfully created ====")

    def create_agents(self) -> list[agt.Agent]:
        """
        """
        raise NotImplementedError

    def create_graphs(self) -> list[gr.Graph]:
        """
        """
        raise NotImplementedError

    def create_groups(self) -> list[grp.Group]:
        """
        """
        raise NotImplementedError

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: A potentially partial list of the model names which have existing saves.
        :type existing_saves: list[str], optional
        """
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate model objects to all relevant models.

        :param missing_saves: A potentially partial list of the model names which are missing their saves.
        :type missing_saves: list[str], optional
        """
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model in the tester class.

        :param missing_saves: A potentially partial list of the model names which are missing their saves.
        :type missing_saves: list[str], optional
        """
        return None


if __name__ == "__main__":
    class TestParameters(TypedDict):
        iterations: int
        model_names: list[str]
        shared_hierarchy_rw: dict[str, tuple[float, float]]
        unique_hierarchy_rw: dict[str, tuple[float, float]]
        shared_relationship_rw: tuple[float, float]
        unique_rel_rw_range: tuple[tuple[float, float], tuple[float, float]]
        hierarchy_names: list[str]
        graph_generation_alg: str

    # The relevant parameters that are being applied in this experiment
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "model_names": ["BASE", "RELS", "HIER", "BOTH"],
        "shared_hierarchy_rw": {
            "A": (0.0, 1.0),
            "B": (0.0, 1.0),
            "C": (0.0, 1.0),
            "D": (0.0, 1.0),
            "E": (0.0, 1.0),
            "F": (0.0, 1.0),
        },
        "unique_hierarchy_rw": {
            "A": (0.0, 1.5),
            "B": (0.0, 0.3),
            "C": (0.0, 0.1),
            "D": (-0.1, 0.2),
            "E": (0.0, 0.01),
            "F": (0.1, 0.2),
        },
        "shared_relationship_rw": (0.0, 0.1),
        "unique_rel_rw_range": (
            (-0.1, 0.1),  # Mean range
            (0.0, 0.5),  # Variance range
        ),
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "graph_generation_alg": "small-world",
    }

    class AgentParameters(TypedDict):
        n_agents: int
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        social_susceptibility: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personality: dict[str, float]
        personal_benefit: dict[bool, float]
        id_base: str

    # The parameters that will be used to create the shared population of agents across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "social_susceptibility": (0.2, 0.8),
        "hierarchy_weighting": (-0.6, 0.6),
        "personality": {
            "social": 0.4,
            "neutral": 0.4,
            "impulsive": 0.2,
        },
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "GEXRW",
    }

    ROOT_DIR: str = "./experiments/GroupBase/RandomWalk"

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "BASE": f"{ROOT_DIR}/RandomWalk_BASE",
        "RELS": f"{ROOT_DIR}/RandomWalk_RELS",
        "HIER": f"{ROOT_DIR}/RandomWalk_HIER",
        "BOTH": f"{ROOT_DIR}/RandomWalk_BOTH",
    }

    # The save paths for each model's logger output (must point to a .csv)
    SAVEFILES: dict[str, str] = {
        "BASE": f"{ROOT_DIR}/BASE_model_variables.csv",
        "RELS": f"{ROOT_DIR}/RELS_model_variables.csv",
        "HIER": f"{ROOT_DIR}/HIER_model_variables.csv",
        "BOTH": f"{ROOT_DIR}/BOTH_model_variables.csv",
    }

    tester: RandomWalkTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(ROOT_DIR))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for model, save_dir in SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(model)
        else:
            directory_missing = True
            missing_savedirs.append(model)

    if directory_missing:
        tester = RandomWalkTester()

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = RandomWalkTester(existing=True)
        tester.load_models()
