import os
from typing import List

import numpy as np
import torch
import torch.nn
import torch.nn.functional
from torch import Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points


class LookupEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.space = self.check_option("space", ["euclidean", "complex"])

        # n3 is only accepted when space is complex
        if self.space == "complex":
            self.regularize = self.check_option("regularize", ["", "lp", "n3"])
        else:
            self.regularize = self.check_option("regularize", ["", "lp"])

        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
        )

        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._maybe_load_user_pretrained()
            self._normalize_embeddings()

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("job.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def _normalize_embeddings(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                self._embeddings.weight.data = torch.nn.functional.normalize(
                    self._embeddings.weight.data, p=self.normalize_p, dim=-1
                )

    @torch.no_grad()
    def _maybe_load_user_pretrained(self) -> None:
        """Overwrite drug entity rows from user.pretrained_path/dim_{dim}/.

        Proteins and unmatched entities keep their random init.
        Relation embedders are skipped.
        """
        if "entity_embedder" not in (self.configuration_key or ""):
            return

        try:
            pretrained_path = self.config.get("user.pretrained_path")
        except KeyError:
            return
        if not pretrained_path:
            return

        dim_dir = os.path.join(pretrained_path, f"dim_{self.dim}")
        emb_path = os.path.join(dim_dir, "embeddings.npy")
        ids_path = os.path.join(dim_dir, "drug_ids.txt")

        if not os.path.isfile(emb_path) or not os.path.isfile(ids_path):
            raise FileNotFoundError(
                f"user.pretrained_path set, but missing files under {dim_dir}. "
                f"Expected embeddings.npy and drug_ids.txt"
            )

        vectors = np.load(emb_path)
        with open(ids_path) as f:
            drug_ids = [line.strip() for line in f if line.strip()]

        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {vectors.shape}")
        if vectors.shape[0] != len(drug_ids):
            raise ValueError(
                f"Row mismatch: {vectors.shape[0]} vectors vs {len(drug_ids)} ids"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Dim mismatch: embeddings have {vectors.shape[1]}, "
                f"lookup_embedder.dim={self.dim}"
            )

        entity_ids = list(self.dataset.entity_ids())
        id_to_index = {eid: i for i, eid in enumerate(entity_ids)}

        row_indices = []
        row_vectors = []
        missing = 0
        for drug_id, vec in zip(drug_ids, vectors):
            idx = id_to_index.get(drug_id)
            if idx is None:
                missing += 1
                continue
            row_indices.append(idx)
            row_vectors.append(vec)

        if not row_indices:
            raise RuntimeError(
                "No drug IDs from pretrained files matched dataset.entity_ids(). "
                "Check that drug_ids.txt uses the same strings as entity_ids.del "
                f"(example drug id: {drug_ids[0]!r})"
            )

        idx_t = torch.tensor(
            row_indices, dtype=torch.long, device=self._embeddings.weight.device
        )
        vec_t = torch.tensor(
            np.asarray(row_vectors),
            dtype=self._embeddings.weight.dtype,
            device=self._embeddings.weight.device,
        )
        self._embeddings.weight.data[idx_t] = vec_t

        self.config.log(
            f"Loaded pretrained drug embeddings from {dim_dir}: "
            f"{len(row_indices)}/{len(drug_ids)} drugs overwritten "
            f"(missing in dataset: {missing}; "
            f"proteins/other entities left at random init)"
        )

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0 and isinstance(job, TrainingJob):
            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self._normalize_embeddings())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self._normalize_embeddings())

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        self._embeddings.weight[
            torch.from_numpy(self_intersect_ind)
            .to(self._embeddings.weight.device)
            .long()
        ] = pretrained_embedder.embed(torch.from_numpy(pretrained_intersect_ind)).to(
            self._embeddings.weight.device
        )

    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._embeddings(indexes.long()))

    def embed_all(self) -> Tensor:
        return self._postprocess(self._embeddings_all())

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        return self._embeddings(
            torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )
        )

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def _abs_complex(self, parameters) -> Tensor:
        parameters_re, parameters_im = (t.contiguous() for t in parameters.chunk(2, dim=1))
        parameters = torch.sqrt(parameters_re ** 2 + parameters_im ** 2 + 1e-14) # + 1e-14 to avoid NaN: https://github.com/lilanxiao/Rotated_IoU/issues/20
        return parameters

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize in ["lp", 'n3']:
            if self.regularize == "n3":
                p = 3
            else:
                p = (
                    self.get_option("regularize_args.p")
                    if self.has_option("regularize_args.p")
                    else 2
                )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                if self.regularize == "n3" and self.space == 'complex':
                    parameters = self._abs_complex(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embeddings(unique_indexes)

                if self.regularize == "n3" and self.space == 'complex':
                    parameters = self._abs_complex(parameters)

                if (p % 2 == 1) and (self.regularize != "n3"):
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
