import json
import random

import polars as pl


class DataSynthesiser:
    def __init__(
        self, response_file: str, output_path: str, community_code: str = "FALSE"
    ):
        self.response_file: str = response_file
        self.output_path: str = output_path
        self.community_code: str = community_code

        self.response_distribution: dict
        with open(self.response_file, "r") as file:
            self.response_distribution = json.load(file)

        self.num_questions: int = len(self.response_distribution.keys())

        self.num_synthetic_entries: int = 0

        self.output_dataframe: pl.DataFrame
        self.output_dict: dict = {"AgentId": []}

        for question in self.response_distribution.keys():
            self.output_dict[question] = []

    def generate_n_entries(self, n: int = 100):
        """
        Using the responses recorded in self.response_distribution, create n randomly distributed
        data entries.

        :param n: The number of randomly distributed data entries to create
        """
        for entry in range(n):
            self.num_synthetic_entries += 1

            agent_id: str = f"{self.community_code}{self.num_synthetic_entries:05}"
            self.output_dict["AgentId"].append(agent_id)

            for question, responses in self.response_distribution.items():
                choices: list[str] = list(responses.keys())
                weights: list[int] = list(responses.values())
                generated_response: str = random.choices(choices, weights=weights, k=1)[
                    0
                ]
                self.output_dict[question].append(generated_response)

    def create_dataframe(self):
        """
        Create a DataFrame from self.output_dict
        """
        self.output_dataframe = pl.DataFrame(self.output_dict)

    def write_csv(self):
        self.output_dataframe.write_csv(self.output_path)
