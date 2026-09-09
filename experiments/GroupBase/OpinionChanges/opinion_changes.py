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
