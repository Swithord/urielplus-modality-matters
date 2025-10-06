import os
import pickle
import pandas as pd
import numpy as np
from typing import Union

from src.latent_islands_typological.functions import compute_vector_distance
from src.speaker_geographic.functions import integrate_geo_data, normalized_w1_distance

GENETIC_DISTANCE_MAX = 32.123024 # Maximum pairwise genetic distance (for normalization)

class DistanceTool:
    """
    A tool to query language distance.
    """
    def __init__(self):
        pass

    def query_distance(self, lang1: str, lang2: str) -> Union[float, np.nan]:
        """
        Queries the distance between two languages.
         Args:
            lang1 (str): The first language code (Glottocode).
            lang2 (str): The second language code (Glottocode).
        Returns:
            float | np.nan: The distance between the two languages, or np.nan if not available.
        """
        raise NotImplementedError


class SpeakerGeographicDistanceTool:
    """
    A tool to query language distance.
    Modality: Geographic
    Representation: Speaker distributions
    """
    def __init__(self, dataset_path: str):
        """
        Initializes the SpeakerGeographicDistanceTool.
        Args:
            dataset_path (str): Path to the per-language, per-country speaker counts CSV file.
        """
        super().__init__()
        self.df = integrate_geo_data(dataset_path)

    def query_distance(self, lang1: str, lang2: str) -> Union[float, np.nan]:
        result = normalized_w1_distance(lang1, lang2, self.df)
        return result


class HyperbolicGeneticDistanceTool:
    """
    A tool to query language distance.
    Modality: Genetic
    Representation: Hyperbolic embeddings
    Note: This tool uses pre-computed distances.
    """
    def __init__(self, dataset_path: str):
        """
        Initializes the HyperbolicGeneticDistanceTool.
        Args:
            dataset_path (str): Path to the precomputed distance matrix CSV file.
        """
        super().__init__()
        self.dist_map = pd.read_csv(dataset_path, index_col=0)

    def query_distance(self, lang1: str, lang2: str) -> Union[float, np.nan]:
        try:
            return self.dist_map.loc[lang1, lang2] / GENETIC_DISTANCE_MAX
        except KeyError:
            return np.nan

class IslandsTypologicalDistanceTool:
    """
    A tool to query language distance.
    Modality: Typological
    Representation: Latent islands model
    """
    def __init__(self, dataset_path: str, islands_path: str):
        """
        Initializes the IslandsTypologicalDistanceTool.
        Args:
            dataset_path (str): Path to the URIEL+ dataset CSV file.
            islands_path (str): Path to the islands pickle file.
        """
        super().__init__()
        self.df = pd.read_csv(dataset_path, index_col=0)

        with open(islands_path, 'rb') as f:
            island_full = pickle.load(f)
        self.islands = island_full

    def query_distance(self, lang1: str, lang2: str) -> Union[float, np.nan]:
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")

        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))

        if not intersection:
            return np.nan
        result = compute_vector_distance(self.islands, self.df.loc[lang1].to_numpy(), self.df.loc[lang2].to_numpy())
        return result
