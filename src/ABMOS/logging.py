class ABMOSLogger:
    """
    The logging module will contain all functions related to logging and/or printing model progress and information
    both during and after simulation
    """

    def __init__(
        self, verbose: bool = False, print_interval: int = 10, write_file: bool = True
    ) -> None:
        """
        :param verbose: a flag to indicate if extended information should be printed during logging
        :param print_interval: the number of model iterations to run in between each printed logging output
        :param write_file: a flag to indicate if a log file should be written to disk at the end of logging
        """
        self.verbose = verbose
        self.print_interval = print_interval
        self.write_file = write_file
