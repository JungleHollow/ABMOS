import argparse

from .agents import Agent, AgentSet
from .graphs import Graph, GraphEdge, GraphNode, GraphSet
from .logging import ABMOSLogger
from .model import ABModel

__version__ = "0.1"
__authors__ = "Manuel Munizaga Sepulveda"
__license__ = "MIT License"
__year__ = "2025"
__repo__ = "https://www.github.com/JungleHollow/ABMOS"

###
# This may be turned into a CLI entry point or extended context manager in the future...
###

parser = argparse.ArgumentParser(
    prog="ABMOS",
    usage="",
    description="An open-source Python package to model social unrest in small yet complex communities",
    epilog="",
)
