from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import TypedDict

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class InfluentialTester:
    """
    A test class that sets up and runs models for a low-influence (LI) scenrario and a high-influence (HI) scenario for
    an experimental comparison.

    In both cases, the models are composed of 5 hierarchies each with unique random walk distributions for their dynamic
    relationships. Additionally, both models will have 40 agents within them, and run for a total of 100 iterations.

    In the case of the LI model, all 40 agents are "low influence" agents with somewhat weaker relationships, and with
    hierarchy graphs having significantly less relationship densitites between agents.

    In the case of the HI model, 36 agents are "low influence" whilst 4 are replaced with "high influence" agents.
    These high influence agents have stronger relationships with their neighbours, and a significantly higher
    relationship density between themselves and their neighbours in the hierarchies.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading existing experiment results.
        :type existing: bool, optional
        """
        # Store the class parameters within the instance
        self.n_agents: int = TEST_PARAMETERS["n_agents"]
        self.n_groups: int = TEST_PARAMETERS["n_groups"]
        self.n_negative: int = TEST_PARAMETERS["n_negative"]

        self.existing: bool = existing

        # Define the data types without assigning values
        self.li_agents: list[agt.Agent]
        self.hi_agents: list[agt.Agent]
        self.li_graphs: list[gr.Graph]
        self.hi_graphs: list[gr.Graph]
        self.li_groups: list[grp.Group]
        self.hi_groups: list[grp.Group]

        # Create the model objects no matter what
        self.li_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            simulate_groups=True,
            save_dir=LI_SAVEDIR,
            data_file=LI_DATAFILE,
            model_id="LI_MODEL",
        )
        self.hi_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            simulate_groups=True,
            save_dir=HI_SAVEDIR,
            data_file=HI_DATAFILE,
            model_id="HI_MODEL",
        )

        # Objects are needed to define and run the models
        if not self.existing:
            # Create the agents
            self.li_agents = self.create_li_agents()
            self.hi_agents = self.create_hi_agents()

            # Create the graphs
            self.li_graphs = self.create_li_graphs(
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                self.li_agents,
            )
            self.hi_graphs = self.create_hi_graphs(
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                self.hi_agents,
            )

            # Create the groups
            self.li_groups = self.create_li_groups()
            self.hi_groups = self.create_hi_groups()

    def create_li_agents(self) -> list[agt.Agent]:
        """
        Creates the population of agents to be used in the LI scenario.

        :return: The generated LI agents.
        :rtype: list[Agent]
        """
        created_li_agents: list[agt.Agent] = []

        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["non_negative_opinion"]
        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        # Define the data types but do not assign any values
        agent_id: str
        agent_opinion: float
        agent: agt.Agent

        created_count: int = 0
        while created_count < self.n_agents - self.n_negative:
            created_count += 1
            agent_id = f"NONN{created_count:04}"

            agent_opinion = rd.uniform(nn_opinion_range[0], nn_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(agent)

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            agent_id = f"NGTV{created_count:04}"

            agent_opinion = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(agent)

        return created_li_agents

    def create_hi_agents(self) -> list[agt.Agent]:
        """
        Creates the population of agents for the HI scenario.

        :return: The generated agent objects for HI.
        :rtype: list[Agent]
        """
        created_hi_agents: list[agt.Agent] = []

        nn_agents: int = self.n_agents - self.n_negative
        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["non_negative_opinion"]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        created_count: int = 0
        while created_count < nn_agents:
            created_count += 1
            nn_agent_id: str = f"NONN{created_count:04}"

            nn_agent_opinion: float = rd.uniform(nn_opinion_range[0], nn_opinion_range[1])

            nn_agent: agt.Agent = agt.Agent(
                nn_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                nn_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(nn_agent)

        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            n_agent_id: str = f"INFN{created_count:04}"

            n_agent_opinion: float = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            n_agent: agt.Agent = agt.Agent(
                n_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                n_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(n_agent)

        return created_hi_agents


if __name__ == "__main__":
    class TestParameters(TypedDict):
        n_agents: int
        n_groups: int
        n_negative: int

    # The parameters set for the tester class itself
    TEST_PARAMETERS: TestParameters = {
        "n_agents": 100,
        "n_negative": 10,
        "n_groups": 10,
    }

    class ModelParameters(TypedDict):
        iterations: int
        silencing_thresh: float
        radical_thresh: float
        negation_thresh: float

    # The model parameters used when creating the ABModel instance
    MODEL_PARAMETERS: ModelParameters = {
        "iterations": 100,
        "silencing_thresh": 0.95,
        "radical_thresh": 0.99,
        "negation_thresh": 0.999,
    }

    # The social hierarchies that will exist in the models
    HIERARCHY_NAMES: list[str] = [
        "family",
        "friends",
        "religion",
        "neighbours",
        "cultural",
    ]

    # The random walk distribution parameters for each hierarchy
    HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
        (0.0, 0.01),  # Family
        (0.0, 0.05),  # Friends
        (0.0, 0.15),  # Religion
        (0.0, 0.08),  # Neighbours
        (0.0, 0.2),  # Cultural
    ]

    # The hierarchy weightings that each agent will assign to the social hierarchies.
    HIERARCHY_WEIGHTINGS: dict[str, float] = {
        "family": 0.9,
        "friends": 0.7,
        "religion": 0.5,
        "neighbours": 0.55,
        "cultural": 0.25,
    }

    class AgentCharacteristics(TypedDict):
        non_negative_opinion: tuple[float, float]
        negative_opinion: tuple[float, float]
        non_influential_connectivity: int
        relationship: tuple[float, float]
        influential_connectivity: int
        influential_relationship: tuple[float, float]
        social_susceptibility: float
        personality: str
        personal_benefit: bool

    # Defining the distributions of the agent characteristics for this experiment
    AGENT_CHARACTERISTICS: AgentCharacteristics = {
        "non_negative_opinion": (0.0, 0.8),
        "negative_opinion": (-1.0, -0.5),
        "non_influential_connectivity": 3,
        "relationship": (-0.4, 0.4),
        "influential_connectivity": 20,
        "influential_relationship": (-0.9, 0.9),
        "social_susceptibility": 0.5,
        "personality": "social",
        "personal_benefit": True,
    }

    # The root directory of this experiment
    ROOT_DIR: str = "./experiments/GroupBase/InfluentialAgents"

    # Define the save directories for each model
    LI_SAVEDIR: str = f"{ROOT_DIR}/InfluentialAgents_LI"
    HI_SAVEDIR: str = f"{ROOT_DIR}/InfluentialAgents_HI"

    # Define the data file paths for each model (must point to a .csv)
    LI_DATAFILE: str = f"{ROOT_DIR}/li_model_variables.csv"
    HI_DATAFILE: str = f"{ROOT_DIR}/hi_model_variables.csv"

    tester: InfluentialTester

    # At least one model save directory does not exist
    if not os.path.exists(LI_SAVEDIR) or not os.path.exists(HI_SAVEDIR):
        # Create the tester normally, setup the models, and begin iterations
        tester = InfluentialTester()
        tester.setup_models()
        tester.run_model_li()
        tester.run_model_hi()
    # Both models exist
    else:
        tester = InfluentialTester(existing=True)
        tester.load_models()
