"""Shared data, embedding, and plotting helpers for English-only clustering."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "quran_data" / "merged_quran.csv"
CLUSTER_DATA_DIR = ROOT_DIR / "data" / "cluster_data"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_english_dataset(path=DATA_PATH):
    dataframe = pd.read_csv(path)
    required_columns = {"surah", "verse", "translation"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    dataframe = dataframe.copy()
    dataframe["translation"] = dataframe["translation"].fillna("").astype(str).str.strip()
    dataframe = dataframe[dataframe["translation"].ne("")].reset_index(drop=True)
    dataframe["verse_id"] = dataframe["surah"].astype(str) + ":" + dataframe["verse"].astype(str)
    return dataframe


def encode_documents(documents, model_name=DEFAULT_MODEL, cache_path=None):
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Install sentence-transformers to generate clustering embeddings."
        ) from error

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def membership_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -(safe_probabilities * np.log(safe_probabilities)).sum(axis=1)


def normalized_membership_entropy(probabilities):
    component_count = probabilities.shape[1]
    if component_count <= 1:
        return np.zeros(probabilities.shape[0])
    return membership_entropy(probabilities) / np.log(component_count)


def save_probability_table(dataframe, probabilities, output_path, prefix):
    probability_columns = [f"{prefix}_{index:02d}" for index in range(probabilities.shape[1])]
    result = dataframe[["surah", "verse", "verse_id", "translation"]].copy()
    result[probability_columns] = probabilities
    result["dominant_membership"] = probabilities.argmax(axis=1)
    result["membership_entropy"] = membership_entropy(probabilities)
    result.to_csv(output_path, index=False, encoding="utf-8")
    return result