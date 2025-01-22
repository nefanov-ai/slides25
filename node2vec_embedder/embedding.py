from node2vec import Node2Vec
import numpy as np


def generate_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4):
    """
    Generate embeddings for each node in the graph using Node2Vec.
    """
    # Initialize Node2Vec
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
    )

    # Train the model
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Get embeddings for all nodes
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings


def aggregate_embeddings(embeddings):
    """
    Aggregate node embeddings into a single tree embedding using mean pooling.
    """
    embedding_matrix = np.array(list(embeddings.values()))
    return np.mean(embedding_matrix, axis=0)

