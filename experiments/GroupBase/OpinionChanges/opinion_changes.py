from __future__ import annotations

import csv
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypedDict

from multiprocessing import Pool
# For type checking
from multiprocessing.pool import Pool as WorkerPool

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md
from gatoh.utils import random_coinflip


class SaveStructDict(TypedDict):
    model_id: str
    max_iterations: int
    change_iteration: int
    changed_agents: dict[str, list[str]]
    changed_groups: list[str]
    current_iteration: int


@dataclass
class SaveStruct:
    """
    Dataclass that is used to store the supporting information from a ModelStruct alongside an ABModel's already defined
    save files.
    """

    # All attributes correspond directly to those in ModelStruct (except for the missing 'model', which is replaced with model ID)
    model_id: str
    max_iterations: int
    change_iteration: int
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    changed_groups: list[str] = field(default_factory=list)
    current_iteration: int = 0

    def __init__(self, struct_to_save: ModelStruct, save_dir: str) -> None:
        """
        Extract all the accompanying information from the given ModelStruct, and then pickle the information to the
        ABModel savedir.

        :param struct_to_save: The structure containing all of the supporting experiment information for a model.
        :type struct_to_save: ModelStruct
        :param save_dir: The model's save directory.
        :type save_dir: str
        """
        self.model_id = struct_to_save.model.model_id
        self.max_iterations = struct_to_save.max_iterations
        self.change_iteration = struct_to_save.change_iteration
        self.changed_agents = deepcopy(struct_to_save.changed_agents)
        self.changed_groups = deepcopy(struct_to_save.changed_groups)
        self.current_iteration = struct_to_save.change_iteration

        # Immediately call self.pickle
        self.pickle_struct(save_dir)

    def pickle_struct(self, save_dir: str) -> None:
        """
        Serialise the accompanying information from the ModelStruct and store it in a pickle file within the
        main ABModel's save directory.

        :param save_dir: The model's save directory.
        :type save_dir: str
        """
        with open(f"{save_dir}/{self.model_id}.pkl", "wb") as pickle_file:
            pickle.dump(self.__dict__, pickle_file)
        return None


@dataclass
class ModelStruct:
    """
    Dataclass that defines a structure to contain all relevant information for handling instance runtimes in this experiment.
    """

    # The ABModel object that has been created for this instance
    model: md.ABModel
    # The maximum number of iterations that the instance will run for
    max_iterations: int
    # The iteration at which the opinion changes will be introduced
    change_iteration: int
    # An <ID : hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies that this occurs in
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    # The Groups whose opinions will be changed due to the agent opinion changes
    changed_groups: list[str] = field(default_factory=list)
    # The current iteration that the instance is at
    current_iteration: int = 0

    def __init__(
        self,
        model: md.ABModel,
        max_iterations: int,
        change_iteration: int,
        changed_agents: dict[str, list[str]],
        changed_groups: list[str],
    ) -> None:
        """
        Store the instance model and the relevant iteration information to be able to introduce the opinion changes during runtime.

        :param model: The model object that has been created for this instance.
        :type model: ABModel
        :param max_iterations: The total number of iterations the model will run for.
        :type max_iterations: int
        :param change_iteration: The iteration during which the opinion changes are introduced.
        :type change_iteration: int
        :param changed_agents: An <ID : hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies that this occurs in.
        :type changed_agents: dict[str, list[str]]
        :param changed_groups: The agents' corresponding groups within which opinion changes will occur.
        :type changed_groups: list[str]
        """
        self.model = model
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.change_iteration = change_iteration
        self.changed_agents = changed_agents
        self.changed_groups = deepcopy(changed_groups)


class OpinionChangesTester:
    """
    A test class that sets up models which are identical in every way, but sudden and strong opinion changes will
    be introduced in random agents at random iterations during their runtime.

    For this experiment, opinion changes will only be introduced once during a model's runtime, using a randomly
    sampled subset of agents, and a mechanism that will generate significantly different opinion values to assign each
    agent, regardless of the initial polarity or magnitude of the opinion.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership, agent group membership,
    and initial conditions; with only the iterations and agents at which opinion chagnes are introduced differing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the experiment is loading past results.
        :type existing: bool, optional
        """
        self.n_agents: int = AGENT_PARAMETERS["n_agents"]
        self.n_groups: int = GROUP_PARAMETERS["n_groups"]
        self.model_intervals: list[int] = list(
            np.linspace(
                0,
                TEST_PARAMETERS["iterations"],
                num=int(
                    TEST_PARAMETERS["iterations"]
                    // TEST_PARAMETERS["opinion_change_interval"]
                ),
                dtype=int,
            )
        )
        self.model_repeats: int = TEST_PARAMETERS["repeats"]

        self.existing: bool = existing
        self.model_saves: dict[str, str] = {}

        # Dynamic model space
        self.models: list[ModelStruct] = []

        # Define the lists that will contain the populations of Agents, Groups, and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []
        self.model_groups: list[grp.Group] = []

        if not self.existing:
            self.create_agents()
            self.create_graphs(self.model_agents)
            self.create_groups(self.model_graphs)
        else:
            self.model_saves = SAVEDIRS
            self.load_agents()
            self.load_graphs()
            self.load_groups()

    def get_struct(self, model_name: str) -> ModelStruct:
        """
        A getter function that iterates over self.models, returning the appropriate ModelStruct object.

        :param model_name: The unique model ID for the model struct's model.
        :type model_name: str
        :raises RuntimeError: If no valid struct is found.
        :return: The model struct containing the model with the specified ID.
        :rtype: ModelStruct
        """
        return_struct: ModelStruct | None = None

        for model_struct in self.models:
            if model_struct.model.model_id == model_name:
                return_struct = model_struct
                break

        if return_struct is None:
            raise RuntimeError(f"Model {model_name} was not found in the tester's instances...")
        else:
            return return_struct

    def create_agents(self) -> None:
        """
        Generates and sets the shared population of Agent objects that will be used across the instances.
        """
        return None

    def pickle_agents(self) -> None:
        """
        Serialises the tester's initial shared Agent population to a subdirectory within the experiment directory.
        """
        return None

    def load_agents(self) -> None:
        """
        Deserialises the tester's initial shared Agent population and loads them into memory.
        """
        return None

    def create_graphs(self, agents: list[agt.Agent]) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.

        :param agents: The population of agents to use for graph creation.
        :type agents: list[Agent]
        """
        return None

    def pickle_graphs(self) -> None:
        """
        Serialises the tester's initial shared Graph population to a subdirectory within the experiment directory.
        """
        return None

    def load_graphs(self) -> None:
        """
        Deserialises the tester's initial shared Graph population and loads it into memory.
        """
        return None

    def create_groups(self, graphs: list[gr.Graph]) -> None:
        """
        Generates and sets the shared collection of Group objects that will be used across the instances.

        :param graphs: The population of social hierarchy graphs to cluster and form groups from.
        :type graphs: list[Graph]
        """
        return None

    def pickle_groups(self) -> None:
        """
        Serialises the tester's initial shared Group population to a subdirectory within the experiment directory.
        """
        return None

    def load_groups(self) -> None:
        """
        Deserialises the tester's initial shared Group population and loads them into memory.
        """
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: A potentially partial list of model names representing models that have existing saves.
        :type existing_saves: list[str], optional
        """
        return None

    def create_savedir_validation(self) -> None:
        """
        Writes a csv file with columns ["model_name", "model_savedir"] containing the relevant information for the models
        of all the instances that have been initialised during model setup.

        This is done in order to allow for checking of missing instance save directories if the tester is being initialised
        from an existing run.
        """
        return None

    def initialise_model_structs(self, missing_saves: list[str] | None = None) -> None:
        """
        Uses the relevant information provided to create appropriate ABModel objects and wrap them in a ModelStruct.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        """
        return None

    def save_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Saves the model objects along with the information contained in their corresponding ModelStruct to allow for future loading.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        """
        return None

    def run_models(self, missing_saves: list[str] | None = None, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs each model instance in the tester class, calling the custom iteration function, and introducing the
        sudden opinion changes at the correct iteration.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        :param worker_pool: A pool of workers that can distribute the processing of the iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        return None

    def agent_opinion_change(self, initial_opinion: float) -> float:
        """
        A helper function that looks at the direction and magnitude of an initial Agent opinion and then
        significantly changes it following a set process.

        :param initial_opinion: The agent's initial opinion.
        :type initial_opinion: float
        :return: An opinion value which is significantly different from the initial one.
        :rtype: float
        """
        raise NotImplementedError

    def custom_iterate(self, model_struct: ModelStruct, worker_pool: WorkerPool | None = None) -> None:
        """
        A custom model iteration function that is able to introduce the opinion changes at the correct iteration
        across instances.

        :param model_struct: A struct containing all relevant information needed to handle the model runtime.
        :type model_struct: ModelStruct
        :param worker_pool: A pool of workers that can distribute the processing of the iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        return None


if __name__ == "__main__":
    MULTIPROCESSED: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSED else None

    class TestParameters(TypedDict):
        iterations: int
        opinion_change_interval: int
        repeats: int
        hierarchy_names: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        relationship_rw: tuple[float, float]
        graph_generation_alg: str
        use_subsetting: bool

    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "opinion_change_interval": 20,
        "repeats": 5,
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
            "E": (0.0, 0.05),
            "F": (0.0, 0.25),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
        "use_subsetting": True,
    }

    class AgentParameters(TypedDict):
        n_agents: int
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personal_benefit: dict[bool, float]
        social_susceptibility: tuple[float, float]
        id_base: str

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "opinions": (-0.9, 0.9),
        "relationships": (-0.9, 0.9),
        "hierarchy_weighting": (-0.75, 0.75),
        "personal_benefit": {True: 0.3, False: 0.7},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "GEXOC"  # (Grouped EXperiment Opinion Changes)
    }

    class GroupParameters(TypedDict):
        n_groups: int

    # The parameters that will be used to create the Group population that is shared across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 20,
    }

    # The root directory of the entire experiment
    ROOT_DIR: str = "./experiments/GroupBase/OpinionChanges"

    # The root of the directory in which each instance's save directory will be located
    # (using a /models subdirectory just for this experiment due to significant increase in number of instances)
    SAVEDIR_ROOT: str = f"{ROOT_DIR}/models"

    # A path to which a validation file will be written -- outlining the model name and save directory that were generated
    # for each instance using the tester initialisation (to allow for checking of missing saves in the future)
    LOGGED_SAVEDIRS: str = f"{ROOT_DIR}/OpinionChanges_logged_savedirs.csv"

    # A <model name : path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    tester: OpinionChangesTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    if not os.path.exists(LOGGED_SAVEDIRS):
        # The tester has not yet been run, or the validation file was removed
        directory_missing = True
    else:
        with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
            csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)
            for row in csv_reader:
                SAVEDIRS[row["model_name"]] = row["model_savedir"]

        for model, save_dir in SAVEDIRS.items():
            dir_name: str = deepcopy(save_dir).split("/")[-1]
            if dir_name in save_dirs:
                existing_savedirs.append(model)
            else:
                directory_missing = True
                missing_savedirs.append(model)

    if directory_missing:
        tester = OpinionChangesTester()

        if len(existing_savedirs) > 0:
            # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.initialise_model_structs(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
        else:
            tester.initialise_model_structs()
            tester.run_models(worker_pool=WORKER_POOL)
    else:
        tester = OpinionChangesTester(existing=True)
        tester.load_models()

    # Ensure that the multiprocessing pool is terminated once all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
