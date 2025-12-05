from __future__ import annotations


class AgentSpace:
    """
    A class that is used to represent the environment in which the Model Agents are interacting with each other.
    """

    def __init__(
        self,
        xlims: tuple[float, float] = (0.0, 100.0),
        ylims: tuple[float, float] = (0.0, 100.0),
        allow_agent_overlap: bool = True,
    ) -> None:
        """
        Initialise the AgentSpace for the Model. By default, a 100x100 2-dimensional space is used.
        """
        self.xlims: tuple[float, float] = xlims
        self.ylims: tuple[float, float] = ylims
        self.allow_agent_overlap: bool = allow_agent_overlap

    def get_limits(self) -> dict[str, tuple[float, float]]:
        """
        A getter method that returns the AgentSpace map limits as an organised dictionary.
        """
        return {"xlims": self.xlims, "ylims": self.ylims}
