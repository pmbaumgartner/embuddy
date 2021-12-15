# EmBuddy

`embuddy` is a package that helps with using text embeddings for local data analysis.

EmBuddy currently supports [models](https://www.sbert.net/docs/pretrained_models.html) available through `SentenceTransformers`.

## Usage

We"ll look at a basic workflow with data from HappyDB.

```python
import pandas as pd
from src.embuddy import EmBuddy

df = pd.read_csv(
    "https://github.com/megagonlabs/HappyDB/raw/master/happydb/data/cleaned_hm.csv",
    nrows=5000,
    usecols=["hmid", "cleaned_hm"],
)

emb = EmBuddy("all-MiniLM-L6-v2")

embeddings = emb.embed(df["cleaned_hm"].tolist())

# do something with `embeddings`...
# project them in UMAP, use them as features in ML, etc
```

Up until now, everything is pretty much the same as using `SentenceTransformers`. Now we'll look at what's unique about EmBuddy.

**Cached Embeddings**

Embeddings are cached within EmBuddy. If we run this again, it should be quite fast!

```python
embeddings = emb.embed(df["cleaned_hm"].tolist())
```

**Approximate Nearest Neighbors Search**

We can use Approximate Nearest Neighbors (ANN) to do semantic search with the embeddings.

=== "Code"

    ``` python
    emb.build_ann_index()
    emb.nearest_neighbors("I made pudding.")
    ```

=== "Result"

    ``` python
    [[(2132, 'I made delicious rice pudding', 0.18222559085871814),
    (2237,
    'I made a tasty chocolate pudding, and every one enjoyed it. ',
    0.21807944021581382),
    (3845, 'I ate delicious rice pudding', 0.269050518351444),
    (2601, 'I made a delicious meal.', 0.3264919047542342),
    (800, 'I made a delicious breakfast.', 0.375725587426599),
    (3262, 'i made a new flavor of homemade ice cream.', 0.38092683794557425),
    (4037, 'I made a really good new recipe for dinner.', 0.3974327179384344),
    (3685, 'I made a tasty breakfast', 0.40264095427046276),
    (4649, 'I made a turkey dinner.', 0.416024635770632),
    (2725,
    'I made delicious rice pudding\r\nI ate delicious rice pudding\r\nI complete 5 Mturk surveys\r\nI taught my dog a new trick\r\nI received a check in the mail',
    0.4244762467642832)]]
    ```

Under the hood, we use [PyNNDescent](https://pynndescent.readthedocs.io/en/latest/index.html) for ANN search. You can pass any `nndescent_kwargs` you'd like to change the underlying ANN search index.

=== "Code"

    ``` python
    emb.build_ann_index(nndescent_kwargs={"metric": "tsss", "n_neighbors": 5})
    emb.nearest_neighbors("I walked my dog.")
    ```

=== "Result"

    ``` python
    [[(3943, 'I went for a walk with my doggie.', 0.06725721),
    (3226, 'Took my dog for a long walk around the neighborhood ', 0.1493121),
    (477, 'I walked my dog and he was licking me. ', 0.15133144),
    (4793, 'I took my dog for a walk, and he was happy.', 0.16377613),
    (1845, 'I took my dog for a long walk on a beautiful day.', 0.18119085),
    (1770, 'I took my dogs for a walk because it was a nice day', 0.1910228),
    (4457, 'I was really happy walking my dog in the park.', 0.23755807),
    (4563,
    'As I was walking to retrieve the mail I saw a neighbor walking their adorable dog. ',
    0.24459706),
    (3994, 'My dog greeted me when I came home. ', 0.24681003),
    (4214, 'I brought my dog to the dog park for the first time. ', 0.2755207)]]
    ```

**Search ANN by Vector**

You can also search by vector. 

=== "Code"

    ``` python
    dog_docs = [doc for doc in df["cleaned_hm"].tolist() if "dog" in doc]
    # Function call is a shortcut for emb.embed
    # You can turn off cache if you don't want to save the embedding.
    mean_dog_vector = emb(dog_docs, cache=False).mean(axis=0)
    similar_docs = emb.nearest_neighbors_vector(mean_dog_vector, k=100)
    most_dog_like_cat_docs = [
        (ix, doc, sim) for (ix, doc, sim) in similar_docs[0] if " cat " in doc
    ]
    ```

=== "Result"

    ``` python
    [(2300, 'I played with my cat throughout the day.', 0.3908988),
    (3884,
    'Went downstairs and cuddled with my cat on the couch watching TV.',
    0.39648643)]
    ```

**Persistence/Serialization**

Done embedding for now but want to add more later?

```python
emb.save("myembeddings.emb")

# some time later... 
my_old_emb = EmBuddy.load("myembeddings.emb")
my_old_emb.nearest_neighbors("I made even better pudding!")
```

**Project to 2D with UMAP**

```python
umap_embeddings = emb.build_umap()

my_old_emb = EmBuddy.load("myembeddings.emb")
umap_embeddings_loaded = my_old_emb.umap_embeddings
```