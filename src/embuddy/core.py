from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import joblib
import joblib_zstd
import numpy as np
import zarr
from pynndescent import NNDescent
from sentence_transformers import SentenceTransformer
from umap import UMAP

from .errors import IndexNotBuiltError, NNDescentHyperplaneError

joblib_zstd.register()


class EmBuddy:
    def __init__(self, model_name: str):
        """A buddy for using text embeddings.

        Args:
            model_name (str): SentenceTransformer model used for embedding
        """
        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.doc_cache: List[str] = []
        self.embedding_cache: np.ndarray = np.empty(
            shape=(0, self.model.get_sentence_embedding_dimension())
        )
        self.ann: Optional[NNDescent] = None
        self.umap_embeddings: Optional[np.ndarray] = None
        self._last_built_len: int = 0

    def embed(self, docs: Union[str, List[str]], cache: bool = True) -> np.ndarray:
        """Embed documents.

        Args:
            docs (Union[str, List[str]]): A string or list of strings to embed
            cache (bool, optional): Whether to cache embedding results. Defaults to True.

        Returns:
            np.ndarray: Embeddings of input documents.
        """
        if isinstance(docs, str):
            docs = [docs]

        if cache:
            result = np.empty(
                shape=(len(docs), self.model.get_sentence_embedding_dimension()),
                dtype=np.float32,
            )
            uncached_docs = list(set([d for d in docs if d not in self.doc_cache]))
            if uncached_docs:
                self.doc_cache.extend(uncached_docs)
                uncached_embeddings = self.model.encode(
                    uncached_docs, convert_to_numpy=True
                )
                self.embedding_cache = np.append(
                    self.embedding_cache, uncached_embeddings, axis=0
                )
                for i, doc in enumerate(docs):
                    if doc in uncached_docs:
                        result[i] = uncached_embeddings[uncached_docs.index(doc)]
                    else:
                        result[i] = self.embedding_cache[self.doc_cache.index(doc)]
            else:
                for i, doc in enumerate(docs):
                    result[i] = self.embedding_cache[self.doc_cache.index(doc)]
        else:
            result = self.model.encode(docs)
        return result

    def __call__(self, docs: Union[str, List[str]], cache: bool = True) -> np.ndarray:
        """Shortcut for [EmBuddy.embed][src.embuddy.core.EmBuddy.embed]"""
        return self.embed(docs=docs, cache=cache)

    def save(self, path: Union[str, Path], overwrite: bool = True) -> zarr.Group:
        """Save the current state of EmBuddy to disk.

        Embeddings and Docs arrays are saved and compressed using zarr.
        The ANN Index, if it exists, is saved using joblib with `zstd` compression.

        Note that this is a directory containing the required data.

        Args:
            path (Union[str, Path]): Location to save EmBuddy data
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to True.

        Returns:
            zarr.Group: Group object containing an `embedding` array of the
            embeddings and a `docs` array of the docments
        """
        if isinstance(path, str):
            path = Path(path)

        g = zarr.group(store=str(path), overwrite=overwrite)
        g.attrs["model_name"] = self.model_name
        g.create_dataset(name="embeddings", data=self.embedding_cache)
        g.create_dataset(name="docs", data=self.doc_cache, dtype=str)
        if self.umap_embeddings is not None:
            g.create_dataset(name="umap_embeddings", data=self.umap_embeddings)
        if self.ann is not None:
            joblib.dump(self.ann, str(path / "ann_index.zstd"), compress=("zstd", 5))
        return g

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmBuddy":
        """Load a previously saved EmBuddy from disk

        Returns:
            EmBuddy: A loaded instance of Embuddy
        """
        if isinstance(path, str):
            path = Path(path)

        g = zarr.open_group(str(path))
        model_name = g.attrs["model_name"]
        embuddy = EmBuddy(model_name)
        embuddy.doc_cache = list(g["docs"])
        embuddy.embedding_cache = g["embeddings"][:]
        umap_check = g.get("umap_embeddings")
        if umap_check:
            embuddy.umap_embeddings = g["umap_embeddings"]
        if (path / "ann_index.zstd").exists():
            embuddy.ann = joblib.load(str(path / "ann_index.zstd"))
        return embuddy

    def build_ann_index(
        self, nndescent_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Builds the Approximate Nearest Neighbors (ANN) index

        Args:
            nndescent_kwargs (Optional[Dict[str, Any]], optional): Optional kwargs to pass to NNDescent. Defaults to None.

        Raises:
            NNDescentHyperplaneError: If ANN can't be built due to small data.
        """
        nndescent_kwargs = _build_nndescent_kwargs_dict(nndescent_kwargs)
        nndescent_kwargs["n_neighbors"] = (
            10 if len(self.doc_cache) < 60 else nndescent_kwargs["n_neighbors"]
        )
        try:
            index = NNDescent(self.embedding_cache, **nndescent_kwargs)
            index.prepare()
            self.ann = index
            self._last_built_len = len(self.doc_cache)
        except ValueError as e:
            if "hyperplane" in str(e):
                raise NNDescentHyperplaneError(
                    "NNDescent cannot build index."
                    " Usually this means data is too small -"
                    " try adding more embeddings and build again."
                )
            else:
                raise e  # pragma: no cover

    def nearest_neighbors(
        self, docs: Union[str, List[str]], k: int = 10
    ) -> List[List[Tuple[int, str, float]]]:
        """Find the nearest neighbors (i.e. most similar) from
         cached docs for the input documents.

        Args:
            docs (Union[str, List[str]]): Docs to find nearest neighbors of.
            k (int, optional): Number of nearest neighbors. Defaults to 10.

        Raises:
            IndexNotBuiltError: If `build_ann_index` has not been run

        Returns:
            List[List[Tuple[int, str, float]]]: For each document, a list of tuples
            containing the document index, document string, and distance for the nearest
            neighbors for each input doc. Sorted by distance.
        """
        if self.ann is None or not isinstance(self.ann, NNDescent):
            raise IndexNotBuiltError(
                "Approximate Nearest Neighbors index not built."
                " Call `build_ann_index` before using this method."
            )
        if self._last_built_len < len(self.doc_cache):
            warn(
                f"{len(self.doc_cache)} embeddings exist in cache, "
                f"but ANN was last built with {self._last_built_len} embeddings. "
                "You are not querying all embeddings. You can rebuild the ANN "
                "index with `build_ann_index`."
            )
        if isinstance(docs, str):
            docs = [docs]

        query_data = self.embed(docs, cache=False).reshape(1, -1)
        neighbors, distances = self.ann.query(query_data, k=k)

        result = []
        for i, doc in enumerate(docs):
            neighbor_ix = neighbors[i]
            neighbor_docs = [self.doc_cache[ix] for ix in neighbor_ix]
            neighbor_dist = distances[i]
            doc_result = []
            for (ix, doc, dist) in zip(neighbor_ix, neighbor_docs, neighbor_dist):
                doc_result.append((ix, doc, dist))
            result.append(doc_result)
        return result

    def nearest_neighbors_vector(
        self, query_vector: np.ndarray, k: int = 10
    ) -> List[List[Tuple[int, str, float]]]:
        """Find the nearest neighbors (i.e. most similar) from
         cached docs of the input vectors.

        You can use this to find similar documents to an arbitrary vector --
        e.g. a vector for a document that doesn't exist, or the mean vector for
        a collection of documents.

        Args:
            query_vector (np.ndarray): An array of vectors
                to find the nearest neighbors for.
            k (int, optional): Number of nearest neighbors. Defaults to 10.

        Raises:
            IndexNotBuiltError: If `build_ann_index` has not been run

        Returns:
            List[List[Tuple[int, str, float]]]: For each vector, a list of tuples
            containing the document index, document string, and distance for the nearest
            neighbors for each input doc. Sorted by distance.
        """
        if self.ann is None or not isinstance(self.ann, NNDescent):
            raise IndexNotBuiltError(
                "Approximate Nearest Neighbors index not built."
                " Call `build_ann_index` before using this method."
            )
        if self._last_built_len < len(self.doc_cache):
            warn(
                f"{len(self.doc_cache)} embeddings exist in cache, "
                f"but ANN was last built with {self._last_built_len} embeddings. "
                "You are not querying all embeddings. You can rebuild the ANN "
                "index with `build_ann_index`."
            )
        query_vector = query_vector.reshape(1, -1)

        neighbors, distances = self.ann.query(query_vector, k=k)

        result = []
        for i, doc in enumerate(query_vector):
            neighbor_ix = neighbors[i]
            neighbor_docs = [self.doc_cache[ix] for ix in neighbor_ix]
            neighbor_dist = distances[i]
            doc_result = []
            for ix, doc, dist in zip(neighbor_ix, neighbor_docs, neighbor_dist):
                doc_result.append((ix, doc, dist))
            result.append(doc_result)
        return result

    def build_umap(
        self, umap_kwargs: Optional[Dict[str, Any]] = None, return_array=True
    ) -> Optional[np.ndarray]:
        """Builds a 2D projection of the embeddings with UMAP
        default settings except metric="cosine".

        Args:
            umap_kwargs (Optional[Dict[str, Any]], optional): Custom UMAP kwargs. Defaults to None.
            return_array (bool, optional): Return the UMAP array. Defaults to True.

        """
        umap_kwargs = _build_umap_kwargs_dict(umap_kwargs)
        umap = UMAP(**umap_kwargs)
        self.umap_embeddings = umap.fit_transform(self.embedding_cache)
        if return_array:
            return self.umap_embeddings
        return None


def _build_umap_kwargs_dict(
    umap_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    defaults = {
        "n_neighbors": 15,
        "n_components": 2,
        "random_state": 1234,
        "metric": "cosine",
        "verbose": True,
    }
    return defaults if umap_kwargs is None else {**defaults, **umap_kwargs}


def _build_nndescent_kwargs_dict(
    nndescent_kwargs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Helper function for creating default NNDescent kwargs.

    Args:
        nndescent_kwargs (Optional[Dict[str, Any]]): Input kwargs.

    Returns:
        Dict[str, Any]: NNDescent default kwargs, if not set in the input.
    """
    defaults = {
        "compressed": True,
        "n_neighbors": 30,
        "random_state": 1234,
        "metric": "cosine",
    }
    return defaults if nndescent_kwargs is None else {**defaults, **nndescent_kwargs}
