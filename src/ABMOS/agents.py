from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Hashable, Iterable
from random import Random
from typing import Any, override

import numpy as np
import polars as pl


class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.
    """

    def __init__(self, *args, **kwargs):
        """
        Supported positional arguments:
            - <dict> of {graph_id : weight} for the personal value that this Agent assigns to each social hierarchy
            - <integer> to set an explicit id for the Agent
            - <float> in the range [-1, 1] to set the Agent's initial opinion on the topic of interest
            - (x, y) for the initial position of the Agent
            - <string> to define the Agent's personality (i.e. "Rational", "Erratic", "Impulsive", etc...)

        :param args: positional arguments that can be passed to each Agent
        :param kwargs: keyword arguments that can be passed to each Agent
        """

        self.id: int | None = None
        self.opinion: float | None = 0.0
        self.social_weightings: dict[str, float] = {}
        self.personality: str = "Neutral"
        self.position: tuple[int, int]

        if args:
            for arg in args:
                match arg:
                    case dict():
                        self.add_attribute("social_weightings", arg)
                    case int():
                        self.add_attribute("id", arg)
                    case float():
                        self.add_attribute("opinion", arg)
                    case tuple():
                        self.add_attribute("position", arg)
                    case str():
                        self.add_attribute("personality", arg)

        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args.
                self.add_attribute(key, value)

    def add_attribute(
        self, name: str, value: Any | None = None, overwrite: bool = False
    ) -> None:
        """
        Dynamically add an attribute to this Agent object.

        :param name: The name of the attribute to be added.
        :param value: Optional initial value of the attribute.
        """
        if not overwrite and name in self.__dict__.keys():
            # Print a warning but do not change any attributes or crash the model if overwriting an existing attribute without meaning to.
            print(
                "Attempting to overwrite an existing Agent attribute without meaning to."
            )
        else:
            self.__dict__[name] = value

    def get_attribute(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            print("Attempting to get an Agent attribute which doesn't exist.")
            return None

    def step(self):
        """
        Step the individual agent object
        """
        pass

    def update_state(self):
        """
        Updates the internal state of the agent after the model has stepped.
        """
        pass

    def radicalisation(self, neighbours: Iterable[Agent]) -> bool:
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param neighbours: A list of all agents that "neighbour" this agent in any model layer.
        """
        match self.__getattribute__("personality"):
            case "rational":
                pass
            case "erratic":
                pass
            case "impulsive":
                pass
            case None:
                pass
        return False  # TODO: Finish this method (returning False to suppress typing warnings)

    def evolve_relationships(self):
        """
        Experimental function that aims to model the constantly evolving relationships between Agents over time
        """
        raise NotImplementedError(
            "Agent relationship evolution has not been implemented as a feature yet."
        )

    def life_events(self):
        """
        Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time
        """
        raise NotImplementedError(
            "Agent life events have not been implemented as a feature yet."
        )

    def __in__(self, iterable: Iterable[Agent]) -> bool:
        for agent in iterable:
            if self == agent:
                return True
        return False


class AgentSet(pl.Series):
    """
    An ordered collection of Agent objects that maintains consistency for the Model
    """

    def __init__(self, agents: Iterable[Agent] = [], random: Random | None = None):
        self.agents = agents
        self._agents = pl.Series(self.agents)
        self.random: Random | None = None
        pass

    def __len__(self) -> int:
        """
        :return: the number of agents present in the AgentSet
        """
        return len(self._agents)

    def __iter__(self) -> Generator[Agent]:
        """
        :return: an iterator which yields each agent in the AgentSet
        """
        return self._agents.__iter__()

    def __contains__(self, agent: Agent) -> bool:
        """
        :param agent: the specific Agent object to check for
        :return: a boolean indicating if the specified Agent object is in the AgentSet
        """
        return self._agents.__contains__(agent)

    def select(
        self,
        filter_func: Callable[[Agent], bool] | None = None,
        inplace: bool = False,
        k: float = np.inf,
    ) -> AgentSet:
        """
        Select a subset of Agent objects from the AgentSet.

        :param filter_func: a function used to filter the Agent objects
        :param inplace: if True, modify the existing AgentSet, otherwise return a new AgentSet
        :param k: the maximum number of Agent objects to include in the subset
        :return: an AgentSet containing a filtered subset of Agents
        """
        pass

    def __getitem__(
        self, item: int | slice
    ) -> pl.Series:  # TODO: Fix the item type mismatch here
        """
        Retrieve an Agent or slice of Agents from the AgentSet.
        :param item: the index or slice for selecting the agents
        :return: the selected agent or slice of agents based on the specified item
        """
        return self._agents.__getitem__(item)

    def add(self, agent: Agent) -> int:
        """
        Add an Agent to the AgentSet.
        :param agent: the Agent object to be added
        """
        for idx, agnt in enumerate(self._agents):
            if agnt is None:
                self._agents[idx] = agent
                return 1
        self._agents.__add__(agent)
        return 1

    def discard(self, agent: Agent) -> int:
        for idx, agnt in enumerate(self._agents):
            if agent == agnt:
                self._agents[idx] = None
                return 1
        return 0

    def remove(self, agent: Agent) -> int:
        for idx, agnt in enumerate(self._agents):
            if agent == agnt:
                self._agents[idx] = None
                return 1
        raise KeyError("Tried to remove an Agent that doesn't exist in the AgentSet")

    @override
    def __getstate__(self) -> dict:
        """
        Retrive the current state of the AgentSet for serialization.
        :return: a dictionary representing the current state of the AgentSet
        """
        return {"agents": list(self._agents), "random": self.random}


class GroupBy:
    """
    Helper class for AgentSet.groupby

    Attributes:
        groups (dict): a dictionary with the group name as key and group as values
    """

    def __init__(self, groups: dict[Any, list | AgentSet]) -> None:
        """
        Initialises the GroupBy instance

        :param groups: a dictionary with the group name as keys and group as values
        """
        self.groups: dict[Any, list | AgentSet] = groups

    def map(self, method: Callable | str, *args, **kwargs) -> dict[Any, Any]:
        pass

    def count(self) -> dict[Any, int]:
        """
        The function will return a count of the Agents within each group
        """
        pass

    def agg(self, attrib_name: str, func: Callable) -> dict[Hashable, Any]:
        """
        The function will aggregate all Agents into hashed subgroups
        """
        pass

    def __iter__(self):
        """
        An override of the __iter__ function to enable the groups to be iterated correctly
        """
        return iter(self.groups.items())

    def __len__(self):
        """
        An override of the __len__ function to enable the length of the groups to be determined correctly
        """
        return len(self.groups)
