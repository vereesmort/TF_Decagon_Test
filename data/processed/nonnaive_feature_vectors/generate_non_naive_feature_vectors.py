#!/usr/bin/env python3
"""Generate PCA/SVD-reduced monopharmacy feature vectors for non-naive LibKGE init.

Builds an n-hot matrix of drug x monopharmacy-side-effect associations, then writes
reduced embeddings for each candidate lookup_embedder.dim used in the paper search
space (32, 64, 128, 256, 512).

This reproduces the missing step described in Lloyd et al. (2024). The original
paper repo referenced ``user.pretrained_path`` but did not ship the generator script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

DEFAULT_DIMS = [32, 64, 128, 256, 512]


def resolve_repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd
    if (cwd / ".." / "data" / "processed").exists():
        return cwd.parent
    if (cwd / ".." / ".." / "data" / "processed").exists():
        return cwd.parent.parent
    raise FileNotFoundError(
        "Could not locate repo root (expected data/processed/). "
        "Run from the repository root or docs/."
    )


def load_monopharmacy_edges(path: Path) -> pd.DataFrame:
    edges = pd.read_csv(path, sep="\t", header=None, names=["drug_id", "relation", "side_effect_id"])
    if not (edges["relation"] == "MonopharmacySideEffect").all():
        raise ValueError(f"Unexpected relation values in {path}")
    return edges[["drug_id", "side_effect_id"]].drop_duplicates()


def build_n_hot_matrix(edges: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    drug_ids = sorted(edges["drug_id"].unique())
    side_effect_ids = sorted(edges["side_effect_id"].unique())

    drug_to_row = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    side_effect_to_col = {side_effect_id: idx for idx, side_effect_id in enumerate(side_effect_ids)}

    rows = [drug_to_row[drug] for drug in edges["drug_id"]]
    cols = [side_effect_to_col[side_effect] for side_effect in edges["side_effect_id"]]
    data = np.ones(len(rows), dtype=np.float32)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(drug_ids), len(side_effect_ids)),
        dtype=np.float32,
    )
    return matrix, drug_ids, side_effect_ids


def reduce_embeddings(
    matrix: sparse.csr_matrix,
    dim: int,
    *,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, float]:
    if dim > min(matrix.shape):
        raise ValueError(
            f"Cannot reduce to dim={dim} for matrix with shape {matrix.shape}. "
            "Choose a smaller dimension."
        )

    if method == "svd":
        reducer = TruncatedSVD(n_components=dim, random_state=random_state)
        embeddings = reducer.fit_transform(matrix)
        explained_variance = float(reducer.explained_variance_ratio_.sum())
    elif method == "pca":
        dense = matrix.toarray()
        reducer = PCA(n_components=dim, random_state=random_state)
        embeddings = reducer.fit_transform(dense)
        explained_variance = float(reducer.explained_variance_ratio_.sum())
    else:
        raise ValueError(f"Unknown method: {method}")

    return embeddings.astype(np.float32), explained_variance


def write_dimension_bundle(
    output_dir: Path,
    dim: int,
    drug_ids: list[str],
    embeddings: np.ndarray,
    *,
    method: str,
    explained_variance: float,
) -> None:
    dim_dir = output_dir / f"dim_{dim}"
    dim_dir.mkdir(parents=True, exist_ok=True)

    np.save(dim_dir / "embeddings.npy", embeddings)

    pd.DataFrame({"drug_id": drug_ids}).to_csv(dim_dir / "drug_ids.txt", index=False, header=False)

    embedding_cols = [f"v{i}" for i in range(embeddings.shape[1])]
    pd.DataFrame(embeddings, columns=embedding_cols).assign(drug_id=drug_ids)[
        ["drug_id", *embedding_cols]
    ].to_csv(dim_dir / "drug_embeddings.tsv", sep="\t", index=False)

    metadata = {
        "dim": dim,
        "method": method,
        "n_drugs": len(drug_ids),
        "explained_variance_ratio_sum": explained_variance,
    }
    (dim_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    repo_root = resolve_repo_root()
    default_mono = repo_root / "data/processed/monopharmacy_edges.tsv"
    default_out = repo_root / "data/graphs/non-naive/feature_vectors"

    parser = argparse.ArgumentParser(
        description="Generate PCA/SVD-reduced monopharmacy init vectors for non-naive training."
    )
    parser.add_argument(
        "--monopharmacy-edges",
        type=Path,
        default=default_mono,
        help=f"Path to monopharmacy_edges.tsv (default: {default_mono})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Embedding dimensions to export (default: 32 64 128 256 512)",
    )
    parser.add_argument(
        "--method",
        choices=["svd", "pca"],
        default="svd",
        help="Reduction method. 'svd' uses TruncatedSVD on the sparse n-hot matrix (default).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the reducer.",
    )
    parser.add_argument(
        "--write-full-features",
        action="store_true",
        help="Also write the full n-hot matrix to full_features.csv in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.monopharmacy_edges.exists():
        raise FileNotFoundError(
            f"Monopharmacy edges not found: {args.monopharmacy_edges}\n"
            "Run data/processed/process_raw_data.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges = load_monopharmacy_edges(args.monopharmacy_edges)
    matrix, drug_ids, side_effect_ids = build_n_hot_matrix(edges)

    print(f"Loaded {len(edges)} monopharmacy associations")
    print(f"n-hot matrix shape: {matrix.shape[0]} drugs x {matrix.shape[1]} side effects")
    print(f"Matrix density: {matrix.nnz / matrix.size:.6f}")

    pd.DataFrame({"side_effect_id": side_effect_ids}).to_csv(
        args.output_dir / "side_effect_ids.txt",
        index=False,
        header=False,
    )

    if args.write_full_features:
        dense = matrix.toarray()
        pd.DataFrame(dense, index=drug_ids, columns=side_effect_ids).to_csv(
            args.output_dir / "full_features.csv"
        )
        print(f"Wrote full n-hot matrix to {args.output_dir / 'full_features.csv'}")

    manifest: dict[str, object] = {
        "source_edges": str(args.monopharmacy_edges.resolve()),
        "method": args.method,
        "random_state": args.random_state,
        "n_drugs": len(drug_ids),
        "n_side_effects": len(side_effect_ids),
        "dims": {},
    }

    for dim in sorted(set(args.dims)):
        embeddings, explained_variance = reduce_embeddings(
            matrix,
            dim,
            method=args.method,
            random_state=args.random_state,
        )
        write_dimension_bundle(
            args.output_dir,
            dim,
            drug_ids,
            embeddings,
            method=args.method,
            explained_variance=explained_variance,
        )
        manifest["dims"][str(dim)] = {
            "path": f"dim_{dim}",
            "explained_variance_ratio_sum": explained_variance,
        }
        print(
            f"dim={dim:>3}  explained_variance={explained_variance:.4f}  "
            f"-> {args.output_dir / f'dim_{dim}'}"
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done. Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
