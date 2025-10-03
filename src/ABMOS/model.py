

class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.
    """
    def __init__(self):
        self.graphs = []
        self.agents = []
        self.iteration = 0

    def init_graphs(self, graphs):
        """
        Defines the graphs to be used for the model.

        :param graphs: list of either created Graph objects, or relevant filenames
        """
        pass

    def init_agents(self, number=100, agents=None):
        """
        Defines the agents to be used for the model.

        :param number: number of agents to be randomly created
        :param agents: list of created Agent objects
        """
        pass

    def step(self):
        """
        Steps the model forward one iteration.
        """
        pass

    def update(self):
        """
        Updates the agents' internal states to match the model step.
        """
        pass

    def visualise(self):
        """
        Visualises the agents' internal states.
        """
        pass
