from __future__ import annotations

import typing

import polars as pl


class DataReader:
    """
    A class that will be used to create and handle an appropriate ABMOS model from the given agent and
    social hierarchy CSV file paths
    """

    def __init__(
        self,
        agents_path: str,
        initial_hierarchies: list[str],
        social_path: str,
        opinions_path: str | None = None,
    ):
        """
        :param agents_path: A relative or absolute file path to a CSV file containing relevant data on model agent characteristics
        :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the reader
        :param social_path: A relative or absolute file path to a CSV file containing relevant data on the relative influence of the existing social hierarchies in the community
        :optional param opinions_path: A relative or absolute file path to a CSV file containing the dependant variables of actual agent opinions; used to compare model accuracy after execution
        """
        self.agents_path: str = agents_path
        self.initial_hierarchies: list[str] = initial_hierarchies
        self.social_path: str = social_path
        self.opinions_path: str | None = opinions_path
