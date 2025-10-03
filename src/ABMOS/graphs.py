

class Graph:
    """
    A graph class that defines a single agent-based model layer.
    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.
    """
    def __init__(self):
        self.nodes = None
        self.edges = None
        self.node_count = 0
        self.edge_count = 0
        self.weights = None
        self.name = None

    def load_graph(self, path):
        """
        Loads a stored Graph object from the given path.

        :param path: path to a stored graph file
        """
        pass

    def create_graph(self, nodes, edges, weights, name=None):
        """
        Creates a new Graph object from the given parameters.

        :param nodes: list of nodes (agents) in the graph
        :param edges: list of edges (relationships) between the agents in the graph
        :param weights: list of weights (relationship strengths) for the edges
        :param name: optional name to label the graph as some specific social hierarchy
        """
        pass

    def change_weights(self, node_1, node_2, value):
        """
        Updates the weight of the relationship between two agents in the graph.
        If no relationship previously exists, a new one is created.

        :param node_1: some agent in the graph
        :param node_2: some other agent in the graph
        :param value: the new weight to assign
        """
        pass

    def remove_node(self, node):
        """
        Removes an agent from the graph, along with any relationships involving it.

        :param node: the node to remove
        """
        pass

    def add_node(self, node, edges, weights):
        """
        Adds a new agent to the graph, and creates all relationships involving it.

        :param node: the Agent object to add
        :param edges: a list of edges between the agent and others in the graph
        :param weights: a list of weights for the corresponding edges
        """
        pass
