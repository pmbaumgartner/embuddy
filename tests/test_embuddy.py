from src.embuddy import __version__, EmBuddy
from src.embuddy.errors import IndexNotBuiltError, NNDescentHyperplaneError
import numpy as np
import pytest


@pytest.fixture
def embuddy_sm():
    return EmBuddy("paraphrase-MiniLM-L3-v2")


def test_version():
    assert __version__ == "0.1.0"


def test_array(embuddy_sm):
    embedding = embuddy_sm.embed(["this is a sentence"])
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 1


def test_single_input_to_list(embuddy_sm):
    embedding = embuddy_sm.embed("this is a sentence")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 1


def test_nocache(embuddy_sm):
    embedding = embuddy_sm.embed(["this is a sentence"], cache=False)
    # did not cache
    assert len(embuddy_sm.doc_cache) == 0
    assert embuddy_sm.embedding_cache.shape[0] == 0


def test_from_cache(embuddy_sm):
    embuddy_sm.embed(["this is a sentence"])
    # do it again, this time from cache
    embuddy_sm.embed(["this is a sentence"])
    assert len(embuddy_sm.doc_cache) == 1
    assert embuddy_sm.embedding_cache.shape[0] == 1


def test_partial_cache(embuddy_sm):
    embuddy_sm.embed(["this is a sentence"])
    # do it again, this time one from cache and one new doc
    embuddy_sm.embed(["this is a sentence", "this is a new sentence"])
    assert len(embuddy_sm.doc_cache) == 2
    assert embuddy_sm.embedding_cache.shape[0] == 2


def test_call(embuddy_sm):
    doc = "this is a sentence"
    assert np.array_equal(embuddy_sm(doc), embuddy_sm.embed(doc))


def test_persist_path(tmp_path, embuddy_sm):
    path = tmp_path / "test_embeddings.emb"
    emb1 = embuddy_sm
    emb1.embed(["this is a sentence"])
    emb1.save(path)

    emb2 = EmBuddy.load(path)
    assert np.array_equal(emb1.embedding_cache, emb2.embedding_cache)


def test_persist_str(tmp_path, embuddy_sm):
    path = str(tmp_path / "test_embeddings.emb")
    emb1 = embuddy_sm
    emb1.embed(["this is a sentence"])
    emb1.save(path)

    emb2 = EmBuddy.load(path)
    assert np.array_equal(emb1.embedding_cache, emb2.embedding_cache)


def test_ann_persist(tmp_path, faker, embuddy_sm):
    # https://faker.readthedocs.io/en/master/pytest-fixtures.html
    texts = faker.texts(nb_texts=61, max_nb_chars=100)
    embuddy_sm.embed(texts)
    embuddy_sm.build_ann_index()
    nn = embuddy_sm.nearest_neighbors(texts[0])

    path = tmp_path / "test_embeddings.emb"
    embuddy_sm.save(path)

    emb2 = EmBuddy.load(path)
    nn2 = emb2.nearest_neighbors(texts[0])
    assert nn == nn2


def test_deduplicate(embuddy_sm):
    embuddy_sm.embed(["this is a sentence"] * 100)
    assert embuddy_sm.embedding_cache.shape[0] == 1


def test_nndescent_kwargs(faker, embuddy_sm):
    texts = faker.texts(nb_texts=61, max_nb_chars=100)
    embuddy_sm.embed(texts)
    nndescent_kwargs = {
        "compressed": True,
        "n_neighbors": 30,
        "random_state": 1234,
    }
    embuddy_sm.build_ann_index(nndescent_kwargs)
    assert all(
        (getattr(embuddy_sm.ann, key) == nndescent_kwargs[key])
        for key in nndescent_kwargs
    )


def test_nn_vector(faker, embuddy_sm):
    texts = faker.texts(nb_texts=61, max_nb_chars=100)
    embeddings = embuddy_sm.embed(texts)
    embuddy_sm.build_ann_index()
    query_vector = embeddings.mean(axis=0)  # column-wise mean value
    k = 10
    neighbors = embuddy_sm.nearest_neighbors_vector(query_vector, k=k)
    assert len(neighbors[0]) == k


def test_nndescent_hyperplane_exception(embuddy_sm):
    texts = ["this sentence", "that sentence"]
    embuddy_sm.embed(texts)
    with pytest.raises(NNDescentHyperplaneError):
        embuddy_sm.build_ann_index()


def test_no_index_exception(embuddy_sm):
    with pytest.raises(IndexNotBuiltError):
        embuddy_sm.nearest_neighbors("Some text")
    with pytest.raises(IndexNotBuiltError):
        embuddy_sm.nearest_neighbors_vector(np.array([1, 2, 3]))


def test_stale_index_warning(faker, embuddy_sm):
    texts = faker.texts(nb_texts=61, max_nb_chars=100)
    embuddy_sm.embed(texts)
    embuddy_sm.build_ann_index()
    newtexts = faker.texts(nb_texts=85, max_nb_chars=100)
    embuddy_sm.embed(newtexts)
    with pytest.warns(UserWarning):
        embuddy_sm.nearest_neighbors("Some text")
    with pytest.warns(UserWarning):
        embuddy_sm.nearest_neighbors_vector(embuddy_sm.embedding_cache[0])
