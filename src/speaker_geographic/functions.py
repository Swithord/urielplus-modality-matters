"""
Geographic language similarity modeling based on speaker population distribution.
"""

import numpy as np
import pandas as pd
from pyproj import Geod
import ot

# Initialize geodesic calculator using WGS84 ellipsoid
geod = Geod(ellps="WGS84")
# Normalization constant (empirically determined)
DISTANCE_MAX = 19999.696911518036


def build_distributions(df):
    """
    Build geographic distributions for each language from dataframe.

    Args:
        df (pd.DataFrame): DataFrame with columns 'glottocode', 'Centroid_Lon',
                          'Centroid_Lat', and 'weight'.

    Returns:
        dict: Dictionary mapping glottocode to tuple of (coordinates, weights).
              Coordinates are numpy array of shape (n, 2) with lon/lat.
              Weights are numpy array of shape (n,) with population weights.
    """
    lang_distributions = {}
    for iso, grp in df.groupby("glottocode"):
        centroid = grp[["Centroid_Lon", "Centroid_Lat"]].to_numpy(dtype="f8")
        weights = grp["weight"].to_numpy(dtype="f8")
        lang_distributions[iso] = (centroid, weights)
    return lang_distributions


def integrate_geo_data(path):
    """
    Load and integrate geographic data for languages.

    Args:
        path (str): Path to CSV file with language geographic data.

    Returns:
        dict: Language distributions mapping glottocode to (coordinates, weights).

    Raises:
        ValueError: If required columns are missing from the data file.
    """
    df = pd.read_csv(path)

    # Ensure the 4 necessary columns exist
    expected = {"glottocode", "Centroid_Lon", "Centroid_Lat", "weight"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise ValueError(
            f"{path} is missing "
            f"columns: {missing}"
        )

    geo_distributions = build_distributions(df)
    return geo_distributions


def geodesic_distance(lon1, lat1, lon2, lat2):
    """
    Calculate geodesic distance between two points on Earth.

    Args:
        lon1 (float): Longitude of first point in degrees.
        lat1 (float): Latitude of first point in degrees.
        lon2 (float): Longitude of second point in degrees.
        lat2 (float): Latitude of second point in degrees.

    Returns:
        float: Distance in kilometers.
    """
    az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2, radians=False)
    return dist / 1000


def cost_matrix_calculate(iso1, iso2, lang_distributions):
    """
    Calculate pairwise geodesic distance matrix between language locations.

    Args:
        iso1 (str): Glottocode of first language.
        iso2 (str): Glottocode of second language.
        lang_distributions (dict): Dictionary of language distributions.

    Returns:
        np.ndarray: Cost matrix of shape (n, m) where n and m are the number
                   of locations for languages iso1 and iso2 respectively.
                   Values are distances in kilometers.
    """
    pts_a, w1 = lang_distributions[iso1]
    pts_b, w2 = lang_distributions[iso2]
    lon_a, lat_a = pts_a[:, 0], pts_a[:, 1]
    lon_b, lat_b = pts_b[:, 0], pts_b[:, 1]

    cost_matrix = np.zeros((len(lon_a), len(lon_b)))

    for i in range(len(lon_a)):
        for j in range(len(lon_b)):
            cost_matrix[i, j] = geodesic_distance(
                lon_a[i], lat_a[i], lon_b[j], lat_b[j]
            )

    return cost_matrix


def w1_distance_language(iso1, iso2, lang_distributions):
    """
    Calculate Wasserstein-1 (Earth Mover's) distance between two languages.

    The W1 distance measures the minimum cost of transforming one language's
    population distribution into another's, using geodesic distance as cost.

    Args:
        iso1 (str): Glottocode of first language.
        iso2 (str): Glottocode of second language.
        lang_distributions (dict): Dictionary of language distributions.

    Returns:
        float: Wasserstein-1 distance in kilometers.
    """
    cost_matrix = cost_matrix_calculate(iso1, iso2, lang_distributions)

    pts_a, w_a = lang_distributions[iso1]
    pts_b, w_b = lang_distributions[iso2]

    return ot.emd2(w_a, w_b, cost_matrix)


def normalized_w1_distance(iso1, iso2, lang_distributions):
    """
    Calculate normalized Wasserstein-1 distance between two languages.

    Args:
        iso1 (str): Glottocode of first language.
        iso2 (str): Glottocode of second language.
        lang_distributions (dict): Dictionary of language distributions.

    Returns:
        float: Normalized distance in range [0, 1], or np.nan if either
               language is not in the distributions.
    """
    if (iso1 not in lang_distributions) or (iso2 not in lang_distributions):
        return np.nan

    lang_dist = w1_distance_language(iso1, iso2, lang_distributions)

    return lang_dist / DISTANCE_MAX
