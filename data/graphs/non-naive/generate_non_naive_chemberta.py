#!/usr/bin/env python3
"""Generate ChemBERTa drug init vectors for non-naive LibKGE training.

Produces the same on-disk layout as the monopharmacy SVD/PCA feature vectors so
LibKGE's patched ``lookup_embedder`` can load them via::

    user:
      pretrained_path: {REPO}/data/graphs/non-naive/chemberta_feature_vectors

Expected layout (``lookup_embedder.dim`` selects the matching folder)::

    chemberta_feature_vectors/
    ├── manifest.json
    ├── dim_32/{embeddings.npy, drug_ids.txt, drug_embeddings.tsv, metadata.json}
    ├── dim_64/...
    ├── dim_128/...
    ├── dim_256/...
    └── dim_512/...

ChemBERTa yields 768-d mean-pooled SMILES embeddings. By default each target dim
is obtained by TruncatedSVD (or PCA). Use ``--no-reduce`` to keep the native
768-d vectors (then set ``lookup_embedder.dim: 768`` in the YAML).

Run from the repository root (``$REPO``), e.g.::

    cd "$REPO"
    # reduced dims (paper search space)
    python data/graphs/non-naive/generate_non_naive_chemberta.py

    # native ChemBERTa size (no SVD/PCA)
    python data/graphs/non-naive/generate_non_naive_chemberta.py --no-reduce

    # or re-embed from SMILES
    python data/graphs/non-naive/generate_non_naive_chemberta.py --recompute-embeddings
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import AutoModel, AutoTokenizer

DEFAULT_DIMS = [32, 64, 128, 256, 512]
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 32
PUBCHEM_BATCH_SIZE = 100


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
        "Run from the repository root ($REPO)."
    )


def stitch_to_cid(stitch_id: str) -> int:
    return int(stitch_id[3:])


def fetch_smiles(cids: Iterable[int], batch_size: int = PUBCHEM_BATCH_SIZE) -> dict[int, str]:
    smiles_by_cid: dict[int, str] = {}
    cid_list = list(cids)

    for start in range(0, len(cid_list), batch_size):
        batch = cid_list[start : start + batch_size]
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{','.join(map(str, batch))}/property/CanonicalSMILES/JSON"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        for record in response.json()["PropertyTable"]["Properties"]:
            smiles = (
                record.get("CanonicalSMILES")
                or record.get("SMILES")
                or record.get("ConnectivitySMILES")
            )
            if smiles:
                smiles_by_cid[record["CID"]] = smiles

        time.sleep(0.2)

    return smiles_by_cid


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)


def embed_smiles(
    smiles_list: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            chunks.append(batch_embeddings.cpu().numpy())

    return np.vstack(chunks).astype(np.float32)


def load_drug_list(repo_root: Path) -> list[str]:
    unique_path = repo_root / "data/processed/unique_drugs.txt"
    if unique_path.exists():
        drugs = [
            line.strip()
            for line in unique_path.read_text().splitlines()
            if line.strip()
        ]
        if drugs:
            return sorted(drugs)

    mono_path = repo_root / "data/processed/monopharmacy_edges.tsv"
    if mono_path.exists():
        edges = pd.read_csv(mono_path, sep="\t", header=None, usecols=[0], names=["drug_id"])
        return sorted(edges["drug_id"].astype(str).unique().tolist())

    raise FileNotFoundError(
        "Need data/processed/unique_drugs.txt or monopharmacy_edges.tsv to list drugs."
    )


def load_or_build_chemberta(
    repo_root: Path,
    *,
    recompute: bool,
    npz_path: Path,
    smiles_cache: Path,
    model_name: str,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    if npz_path.exists() and not recompute:
        data = np.load(npz_path, allow_pickle=True)
        drug_ids = [str(x) for x in data["stitch_ids"].tolist()]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        print(f"Loaded cached ChemBERTa embeddings from {npz_path}")
        print(f"  {len(drug_ids)} drugs, shape {embeddings.shape}")
        return drug_ids, embeddings

    drug_ids = load_drug_list(repo_root)
    drug_records = pd.DataFrame({"stitch_id": drug_ids})
    drug_records["cid"] = drug_records["stitch_id"].map(stitch_to_cid)
    drug_records["smiles"] = pd.Series(pd.NA, index=drug_records.index, dtype="string")

    if smiles_cache.exists():
        cached = pd.read_csv(smiles_cache, dtype={"stitch_id": "string", "smiles": "string"})
        if "cid" not in cached.columns:
            cached["cid"] = cached["stitch_id"].map(stitch_to_cid)
        drug_records = drug_records.drop(columns=["smiles"]).merge(
            cached[["stitch_id", "cid", "smiles"]],
            on=["stitch_id", "cid"],
            how="left",
        )

    missing_mask = drug_records["smiles"].isna()
    if missing_mask.any():
        print(f"Fetching SMILES for {int(missing_mask.sum())} drugs from PubChem…")
        fetched = fetch_smiles(drug_records.loc[missing_mask, "cid"].tolist())
        drug_records.loc[missing_mask, "smiles"] = drug_records.loc[missing_mask, "cid"].map(
            fetched
        )
        smiles_cache.parent.mkdir(parents=True, exist_ok=True)
        drug_records[["stitch_id", "cid", "smiles"]].to_csv(smiles_cache, index=False)

    still_missing = int(drug_records["smiles"].isna().sum())
    if still_missing:
        raise ValueError(f"Missing SMILES for {still_missing} drugs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding {len(drug_records)} SMILES with {model_name} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    drug_records = drug_records.sort_values("stitch_id").reset_index(drop=True)
    embeddings = embed_smiles(
        drug_records["smiles"].tolist(),
        model,
        tokenizer,
        batch_size=batch_size,
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        stitch_ids=drug_records["stitch_id"].to_numpy(),
        cids=drug_records["cid"].to_numpy(),
        smiles=drug_records["smiles"].to_numpy(),
        embeddings=embeddings,
    )
    print(f"Saved raw ChemBERTa npz to {npz_path}")

    return drug_records["stitch_id"].astype(str).tolist(), embeddings


def reduce_embeddings(
    matrix: np.ndarray,
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
        reducer = PCA(n_components=dim, random_state=random_state)
        embeddings = reducer.fit_transform(matrix)
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
    source_dim: int,
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
        "source": "chemberta",
        "source_dim": source_dim,
        "n_drugs": len(drug_ids),
        "explained_variance_ratio_sum": explained_variance,
    }
    (dim_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    repo_root = resolve_repo_root()
    parser = argparse.ArgumentParser(
        description="Generate ChemBERTa init vectors (LibKGE pretrained_path layout)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data/graphs/non-naive/chemberta_feature_vectors",
        help="Directory to pass as user.pretrained_path (default: …/chemberta_feature_vectors)",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=repo_root / "data/processed/drug_chemberta_embeddings.npz",
        help="Cache of full 768-d ChemBERTa embeddings.",
    )
    parser.add_argument(
        "--smiles-cache",
        type=Path,
        default=repo_root / "data/processed/drug_smiles.csv",
        help="CSV cache of stitch_id / cid / smiles.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Target dims matching lookup_embedder.dim "
            f"(default: {DEFAULT_DIMS}; with --no-reduce: native ChemBERTa width only)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["svd", "pca"],
        default="svd",
        help="How to reduce ChemBERTa vectors to each target dim (ignored with --no-reduce).",
    )
    parser.add_argument(
        "--no-reduce",
        action="store_true",
        help=(
            "Write the original ChemBERTa embedding size (typically dim_768) with no "
            "SVD/PCA. Requires lookup_embedder.dim to match that width."
        ),
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignore existing npz and re-run ChemBERTa on SMILES.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    drug_ids, chemberta = load_or_build_chemberta(
        repo_root,
        recompute=args.recompute_embeddings,
        npz_path=args.npz_path,
        smiles_cache=args.smiles_cache,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    # Stable row order for LibKGE matching
    order = np.argsort(drug_ids)
    drug_ids = [drug_ids[i] for i in order]
    chemberta = chemberta[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dim = int(chemberta.shape[1])

    if args.no_reduce:
        dims = args.dims if args.dims is not None else [source_dim]
        if dims != [source_dim]:
            raise ValueError(
                f"--no-reduce writes the native width only (got --dims {dims}, "
                f"source_dim={source_dim}). Omit --dims or pass --dims {source_dim}."
            )
        reduction_method = "none"
    else:
        dims = args.dims if args.dims is not None else DEFAULT_DIMS
        reduction_method = args.method

    manifest: dict[str, object] = {
        "source": "chemberta",
        "model_name": args.model_name,
        "npz_path": str(args.npz_path.resolve()),
        "reduction_method": reduction_method,
        "random_state": args.random_state,
        "n_drugs": len(drug_ids),
        "source_dim": source_dim,
        "dims": {},
    }

    for dim in sorted(set(dims)):
        if args.no_reduce or dim == source_dim:
            embeddings = chemberta.astype(np.float32, copy=True)
            explained_variance = 1.0
            method = "none"
        else:
            embeddings, explained_variance = reduce_embeddings(
                chemberta,
                dim,
                method=args.method,
                random_state=args.random_state,
            )
            method = args.method

        write_dimension_bundle(
            args.output_dir,
            dim,
            drug_ids,
            embeddings,
            method=method,
            explained_variance=explained_variance,
            source_dim=source_dim,
        )
        manifest["dims"][str(dim)] = {
            "path": f"dim_{dim}",
            "explained_variance_ratio_sum": explained_variance,
            "method": method,
        }
        print(
            f"dim={dim:>3}  method={method}  explained_variance={explained_variance:.4f}  "
            f"-> {args.output_dir / f'dim_{dim}'}"
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done. Point user.pretrained_path at:\n  {args.output_dir.resolve()}")
    if args.no_reduce:
        print(f"Set lookup_embedder.dim: {source_dim} in your training YAML.")


if __name__ == "__main__":
    main()
