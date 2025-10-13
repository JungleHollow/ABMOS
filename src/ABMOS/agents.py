from __future__ import annotations
from typing import List, Dict, Optional, Union, Iterable, Any


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
