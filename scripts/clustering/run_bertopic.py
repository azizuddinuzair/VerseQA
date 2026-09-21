"""Run a tunable BERTopic experiment on English Quran translations."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CLUSTER_DATA_DIR,
    DATA_PATH,
    DEFAULT_MODEL,
    encode_documents,
    load_english_dataset,
    membership_entropy,
    normalized_membership_entropy,
)


SCRIPTURAL_STOPWORDS = {
    "and", "the", "you", "of", "to", "is", "are", "was", "were", "they", "them",
    "he", "she", "his", "her", "said", "say", "says", "indeed", "o",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--nr-topics", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-stopwords", action="store_true")
    return parser.parse_args()


def save_distribution_table(dataframe, distributions, output_path, prefix):
    result = dataframe[["surah", "verse", "verse_id", "translation"]].copy()
    columns = [f"{prefix}_{index:02d}" for index in range(distributions.shape[1])]
    result[columns] = distributions
    result["dominant_topic"] = distributions.argmax(axis=1)
    result["membership_entropy"] = membership_entropy(distributions)
    result["normalized_membership_entropy"] = normalized_membership_entropy(distributions)
    result.to_csv(output_path, index=False, encoding="utf-8")
    return result


def save_entropy_plot(dataframe, output_path, title):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(dataframe["normalized_membership_entropy"], bins=30)
    axis.set(xlabel="Normalized membership entropy", ylabel="Verses", title=title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as error:
        raise SystemExit(
            "BERTopic dependencies are missing. Install them with "
            "pip install -r requirements-clustering.txt"
        ) from error

    output_dir = CLUSTER_DATA_DIR / "BERTopic"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    dataframe = load_english_dataset(args.input)
    documents = dataframe["translation"].tolist()
    embeddings = encode_documents(documents, args.model, output_dir / "english_embeddings.npy")
    stopwords = None if args.keep_stopwords else sorted(SCRIPTURAL_STOPWORDS)
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), min_df=3, max_df=0.95)
    topic_model = BERTopic(
        embedding_model=SentenceTransformer(args.model),
        vectorizer_model=vectorizer,
        umap_model=UMAP(
            n_neighbors=args.neighbors,
            n_components=5,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
            prediction_data=True,
        ),
        nr_topics=args.nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, native_probabilities = topic_model.fit_transform(documents, embeddings)
    approximate_distributions, _ = topic_model.approximate_distribution(documents, use_embedding_model=True)
    approximate_distributions = np.asarray(approximate_distributions)
    if approximate_distributions.ndim != 2:
        raise RuntimeError("BERTopic did not return a two-dimensional approximate distribution matrix")

    save_distribution_table(dataframe, approximate_distributions, output_dir / "approximate_distributions.csv", "topic_probability")
    if native_probabilities is not None:
        native_probabilities = np.asarray(native_probabilities)
        save_distribution_table(dataframe, native_probabilities, output_dir / "native_probabilities.csv", "topic_probability")
    pd.DataFrame({
        "surah": dataframe["surah"],
        "verse": dataframe["verse"],
        "verse_id": dataframe["verse_id"],
        "native_topic": topics,
    }).to_csv(output_dir / "native_topics.csv", index=False)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)
    topic_model.save(
        output_dir / "bertopic_model",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )

    visible_topics = topic_info[topic_info["Topic"] >= 0]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(visible_topics["Topic"].astype(str), visible_topics["Count"])
    axis.set(xlabel="Topic", ylabel="Verses", title="BERTopic topic sizes")
    axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(plot_dir / "topic_sizes.png", dpi=160)
    plt.close(figure)

    approximate_table = save_distribution_table(
        dataframe,
        approximate_distributions,
        output_dir / "approximate_distributions.csv",
        "topic_probability",
    )
    save_entropy_plot(approximate_table, plot_dir / "approximate_entropy.png", "BERTopic approximate membership granularity")

    metadata = {
        "input": str(args.input),
        "embedding_model": args.model,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "neighbors": args.neighbors,
        "min_dist": args.min_dist,
        "nr_topics": args.nr_topics,
        "seed": args.seed,
        "stopwords_enabled": not args.keep_stopwords,
        "stopwords": sorted(stopwords) if stopwords is not None else [],
        "vectorizer_ngram_range": [1, 2],
        "vectorizer_min_df": 3,
        "vectorizer_max_df": 0.95,
        "verses": len(dataframe),
        "topic_count": len(visible_topics),
        "native_probability_shape": list(native_probabilities.shape) if native_probabilities is not None else None,
        "approximate_probability_shape": list(approximate_distributions.shape),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved BERTopic results for {len(dataframe)} verses to {output_dir}")


if __name__ == "__main__":
    main()
