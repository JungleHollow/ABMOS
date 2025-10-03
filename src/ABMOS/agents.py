

class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.
    """
    def __init__(self):
        self.name = None
        self.age = None
        self.profession = None
        self.negative_opinion = 0.0
        self.radicalised = False

    def update_state(self):
        """
        Updates the internal state of the agent after the model has stepped.
        """
        pass

    def radicalisation(self, neighbours):
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param neighbours: a list of all agents that "neighbour" this agent in any model layer.
        """
        pass
