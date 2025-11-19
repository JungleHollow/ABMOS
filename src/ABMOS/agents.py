import numpy as np
from __future__ import annotations
from typing import Dict, Optional, Union, Any
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random


class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.
    """
    def __init__(self, argv: Optional[Dict[str, Any]] = None):
        if argv:  # Attributes provided as dict on creation
            for key, val in argv.items():
                self.__dict__[key] = val

    def add_attribute(self, name: str, value: Optional[Any] = None) -> None:
        """
        Dynamically add an attribute to this Agent object.

        :param name: The name of the attribute to be added.
        :param value: Optional initial value of the attribute.
        """
        self.__dict__[name] = value

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
        pass
    
    
class AgentSet(MutableSet, Sequence):
    """
    An ordered collection of Agent objects that maintains consistency for the Model
    """
    def __init__(self, agents: Iterable[Agent], random: Random | None = None):
        self._agents = None
        pass
    
    def __len__(self) -> int:
        """
        :return: the number of agents present in the AgentSet
        """
        return len(self._agents)
    
    def __iter__(self) -> Iterator[Agent]:
        """
        :return: an iterator which yields each agent in the AgentSet 
        """
        return self._agents.keys()
    
    def __contains__(self, agent: Agent) -> bool:
        """
        :param agent: the specific Agent object to check for
        :return: a boolean indicating if the specified Agent object is in the AgentSet
        """
        return agent in self._agents
    
    def select(self, filter_func: Callable[[Agent], bool] | None = None, inplace: bool = False, k: int = np.inf) -> AgentSet:
        """
        Select a subset of Agent objects from the AgentSet.
        
        :param filter_func: a function used to filter the Agent objects
        :param inplace: if True, modify the existing AgentSet, otherwise return a new AgentSet
        :param k: the maximum number of Agent objects to include in the subset
        :return: an AgentSet containing a filtered subset of Agents
        """
        pass
    