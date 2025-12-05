from __future__ import annotations

from .model import ABModel


class AgentSpace:
    """
    A class that is used to represent the environment in which the Model Agents are interacting with each other.
    """

    def __init__(
        self,
        model: ABModel,
        xlims: tuple[float, float] = (0.0, 100.0),
        ylims: tuple[float, float] = (0.0, 100.0),
        allow_agent_overlap: bool = True,
        max_agents_per_grid: int = 4,
        agent_sphere_of_influence: float = 1.0,
        space_type: str = "discrete",
    ) -> None:
        """
        Initialise the AgentSpace for the Model. By default, a 100x100 2-dimensional space is used.

        :param model: The parent ABModel object that the AgentSpace object is being attached to
        :param xlims: The lower and upper bounds of the space's x-axis
        :param ylims: The lower and upper bounds of the space's y-axis
        :param allow_agent_overlap: Flag whether more than one Agent can be in the same grid simultaneously
        :param max_agents_per_grid: The maximum number of Agents that can be in the same grid simultaneously
        :param agent_sphere_of_influence: The maximum distance between agents at which inter-Agent interaction can occur
        :param space_type: The type of space that is used for the Model representation. Current supported types are:
            - "discrete" -- A 2-dimensional (n x m) grid space where n is the size of the x-axis and m is the size of the y-axis
            - "continuous" -- A 2-dimensional gridless space in which Agents can be located at any arbitrary point within the axis limits
        """
        self.parent_model: ABModel = model
        self.xlims: tuple[float, float] = xlims
        self.ylims: tuple[float, float] = ylims
        self.allow_agent_overlap: bool = allow_agent_overlap
        self.max_agents_per_grid: int = max_agents_per_grid
        self.agent_sphere_of_influence: float = agent_sphere_of_influence
        self.space_type: str = space_type

    def get_limits(self) -> dict[str, tuple[float, float]]:
        """
        A getter method that returns the AgentSpace map limits as an organised dictionary.
        """
        return {"xlims": self.xlims, "ylims": self.ylims}
