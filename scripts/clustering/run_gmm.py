"""Run fixed-K, multi-seed GMM experiments on English Quran translations."""

import argparse
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from common import (
    CLUSTER_DATA_DIR,
    DATA_PATH,
    DEFAULT_MODEL,
    encode_documents,
    load_english_dataset,
    normalized_membership_entropy,
    save_probability_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--components", nargs="+", type=int, default=[8, 12, 16, 20])
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--pca-dimensions", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def save_run_audit(dataframe, probabilities, output_path):
    rows = []
    component_count = probabilities.shape[1]
    for component in range(component_count):
        order = np.argsort(probabilities[:, component])[::-1]
        top_indices = order[:10]
        core_indices = order[probabilities[order, component] >= 0.8][:10]
        for audit_type, indices in (("top_posterior", top_indices), ("core_probability_ge_0.8", core_indices)):
            for rank, index in enumerate(indices, start=1):
                rows.append({
                    "audit_type": audit_type,
                    "component": component,
                    "rank": rank,
                    "verse_id": dataframe.iloc[index]["verse_id"],
                    "surah": dataframe.iloc[index]["surah"],
                    "verse": dataframe.iloc[index]["verse"],
                    "probability": probabilities[index, component],
                    "translation": dataframe.iloc[index]["translation"],
                })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def summarize_run(probabilities, model, component_count, seed, reduced_embeddings):
    dominant = probabilities.argmax(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    entropy = normalized_membership_entropy(probabilities)
    return {
        "components": component_count,
        "seed": seed,
        "bic": model.bic(reduced_embeddings),
        "aic": model.aic(reduced_embeddings),
        "verses": len(probabilities),
        "mean_normalized_entropy": entropy.mean(),
        "mean_top1_probability": sorted_probabilities[:, -1].mean(),
        "mean_top2_probability": sorted_probabilities[:, -2].mean(),
        "mean_top1_top2_margin": (sorted_probabilities[:, -1] - sorted_probabilities[:, -2]).mean(),
        "smallest_component": np.bincount(dominant, minlength=component_count).min(),
        "largest_component": np.bincount(dominant, minlength=component_count).max(),
    }


def save_summary_plots(summary, output_dir):
    summary = pd.DataFrame(summary)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for component_count, group in summary.groupby("components"):
        axes[0].plot(group["seed"], group["mean_normalized_entropy"], marker="o", label=f"K={component_count}")
        axes[1].plot(group["seed"], group["mean_top1_top2_margin"], marker="o", label=f"K={component_count}")
    axes[0].set(xlabel="Seed", ylabel="Mean normalized entropy", title="GMM softness")
    axes[1].set(xlabel="Seed", ylabel="Mean top-1 minus top-2", title="GMM certainty")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "softness_by_seed.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = CLUSTER_DATA_DIR / "GMM"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    dataframe = load_english_dataset(args.input)
    embeddings = encode_documents(
        dataframe["translation"].tolist(),
        model_name=args.model,
        cache_path=output_dir / "english_embeddings.npy",
    )
    dimensions = min(args.pca_dimensions, embeddings.shape[0], embeddings.shape[1])
    reduced_embeddings = PCA(n_components=dimensions, whiten=True, random_state=42).fit_transform(embeddings)
    all_summary = []
    labels_by_configuration = {}

    for component_count in args.components:
        for seed in seeds:
            run_dir = output_dir / "runs" / f"k_{component_count:02d}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model = GaussianMixture(
                n_components=component_count,
                covariance_type="diag",
                random_state=seed,
                n_init=args.n_init,
            )
            model.fit(reduced_embeddings)
            probabilities = model.predict_proba(reduced_embeddings)
            labels_by_configuration[(component_count, seed)] = model.predict(reduced_embeddings)
            save_probability_table(dataframe, probabilities, run_dir / "verse_memberships.csv", "component_probability")
            save_run_audit(dataframe, probabilities, run_dir / "representative_verses.csv")
            all_summary.append(summarize_run(probabilities, model, component_count, seed, reduced_embeddings))
            (run_dir / "metadata.json").write_text(json.dumps({
                "input": str(args.input),
                "embedding_model": args.model,
                "components": component_count,
                "seed": seed,
                "n_init": args.n_init,
                "pca_dimensions": dimensions,
                "covariance_type": "diag",
                "verses": len(dataframe),
            }, indent=2), encoding="utf-8")

    stability_rows = []
    for component_count in args.components:
        configurations = [(component_count, seed) for seed in seeds]
        for left, right in combinations(configurations, 2):
            stability_rows.append({
                "components": component_count,
                "seed_left": left[1],
                "seed_right": right[1],
                "adjusted_rand_index": adjusted_rand_score(labels_by_configuration[left], labels_by_configuration[right]),
            })
    pd.DataFrame(all_summary).to_csv(output_dir / "multi_seed_summary.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output_dir / "stability_summary.csv", index=False)
    save_summary_plots(all_summary, plot_dir)
    (output_dir / "experiment_metadata.json").write_text(json.dumps({
        "input": str(args.input),
        "embedding_model": args.model,
        "components": args.components,
        "seeds": seeds,
        "n_init": args.n_init,
        "pca_dimensions": dimensions,
        "covariance_type": "diag",
        "verses": len(dataframe),
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(all_summary)} GMM runs to {output_dir}")


if __name__ == "__main__":
    main()
