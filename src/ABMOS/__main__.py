from .Graphs import Graph
from .Agents import Agent
from .Model import ABModel
import argparse

VERSION = "0.1"
AUTHORS = "Manuel Munizaga Sepulveda"
CONTACT = "manuel.munizaga@pm.me"
LICENSE = "MIT License"
YEAR = "2025"
REPO = "https://www.github.com/JungleHollow/ABMOS"

arg_parser = argparse.ArgumentParser(
    prog="ABMOS",
    description="An open-source Python library aimed at constructing and running agent-based models of community sentiment"
)

arg_parser.add_argument("--no-save", action="store_true", default=False, help="Do not save the final model after simulation is finished.")
arg_parser.add_argument("seed", action="store", type=int, nargs="?", default=None, const=45, help="The random seed to use when running the simulations.")
arg_parser.add_argument("savepath", action="append", nargs="?", default=["./saved_models/"], const="saved_model_XXXX",
                        help="The filename to save the final model to. By default, this will store the models to the relative path: './saved_models/<filename>'.")
arg_parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Execute the runtime with verbose output.")
arg_parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {VERSION}")
arg_parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Execute the runtime with minimal output.")
