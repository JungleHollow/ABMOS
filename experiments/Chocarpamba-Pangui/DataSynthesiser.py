import argparse
import json
import random

import polars as pl


class DataSynthesiser:
    def __init__(
        self,
        response_file: str,
        output_path: str,
        survey_values: str,
        community_code: str = "FALSE",
        social_graphs: list[str] = ["Age", "Family", "Friends", "Religion", "Cultural"],
    ):
        self.response_file: str = response_file
        self.output_path: str = output_path
        self.values_path: str = survey_values
        self.community_code: str = community_code

        self.social_graphs: list[str] = social_graphs

        self.survey_values: dict
        with open(self.values_path, "r") as file:
            self.survey_values = json.load(file)

        self.response_distribution: dict
        with open(self.response_file, "r") as file:
            self.response_distribution = json.load(file)

        self.num_questions: int = len(self.response_distribution.keys())

        self.num_synthetic_entries: int = 0

        self.output_dataframe: pl.DataFrame
        self.output_dict: dict = {"AgentId": []}

        self.output_relationships: dict = {
            hierarchy: [] for hierarchy in self.social_graphs
        }  # Will be lists of (start_node, end_node, weight) triples

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

            is_religious: bool = False
            participates_community: bool = False

            for question, responses in self.response_distribution.items():
                choices: list[str] = list(responses.keys())
                weights: list[int] = list(responses.values())
                generated_response: str = ""

                match question:
                    case "Q11":
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]
                        if generated_response == "Yes":
                            is_religious = True
                    case "Q12":
                        if is_religious:
                            generated_response = random.choices(
                                choices[:-1], weights=weights[:-1], k=1
                            )[0]
                        else:
                            generated_response = "Not important"
                    case "Q27":
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]
                        if generated_response == "Yes":
                            participates_community = True
                    case "Q30":
                        if participates_community:
                            generated_response = random.choices(
                                choices[:2], weights=weights[:2], k=1
                            )[0]
                        else:
                            generated_response = random.choices(
                                choices[2:], weights=weights[2:], k=1
                            )[0]
                    case _:
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]

                self.output_dict[question].append(generated_response)
        self.create_dataframe()
        self.generate_relationships()

    def generate_relationships(self):
        """
        Randomly generate the social hierarchy relationships for the agents based on assumptions and other responses
        """

        for i in self.output_dataframe.iter_rows(named=True):
            for hierarchy in self.social_graphs:
                weight: float  # define but don't initialise the weight variable
                
                match hierarchy:
                    case "Age":
                        for j in self.output_dataframe.iter_rows(named=True):
                            if j["AgentId"] == i["AgentId"]:
                                continue
                            else:
                                weight = 1.0 - (0.25 * abs(self.survey_values["Q1"][i["Q1"]] - self.survey_values["Q1"][j["Q1"]]))
                                self.output_relationships["Age"].append((i["AgentId"], j["AgentId"], weight))
                    case "Family":
                        pass
                    case "Friends":
                        pass
                    case "Religion":
                        if i["Q11"] == "No":
                            continue
                        else:
                            for j in self.output_dataframe.iter_rows(named=True):
                                if j["Q11"] == "No":
                                    continue
                                else:
                                    weight = (
                                        self.survey_values["Q12"][i["Q12"]] + 
                                        self.survey_values["Q13"][i["Q13"]] +
                                        self.survey_values["Q14"][i["Q14"]] +
                                        self.survey_values["Q15"][i["Q15"]] +
                                        self.survey_values["Q16"][i["Q16"]]
                                    ) / 11.0  # 11.0 is the maximum possible score for these questions
                                    self.output_relationships["Religion"].append((i["AgentId"], j["AgentId"], weight))
                    case "Cultural":
                        pass
                    case _:
                        pass

    def create_dataframe(self):
        """
        Create a DataFrame from self.output_dict
        """
        self.output_dataframe = pl.DataFrame(self.output_dict)

    def write_csv(self):
        self.output_dataframe.write_csv(self.output_path)


class SynthesiserArgParser:
    def __init__(self):
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="ABMOS DataSynthesiser",
            description="Create synthetic data from observed distributions for use with the ABMOS library",
        )
        self.parser.add_argument("response_file", type=str)
        self.parser.add_argument("output_path", type=str)
        self.parser.add_argument("-c", "--community_code", default="FALSE", type=str)
        self.parser.add_argument("-n", "--num_entries", default=100, type=int)
        self.parser.add_argument(
            "-s",
            "--social_graphs",
            default=["Age", "Family", "Friends", "Religion", "Cultural"],
            type=list,
        )
        self.main()

    def main(self):
        args: argparse.Namespace = self.parser.parse_args()
        data_synthesiser: DataSynthesiser = DataSynthesiser(
            args.response_file,
            args.output_path,
            args.community_code,
            args.social_graphs,
        )
        data_synthesiser.generate_n_entries(args.num_entries)
        data_synthesiser.write_csv()


if __name__ == "__main__":
    parser: SynthesiserArgParser = SynthesiserArgParser()
