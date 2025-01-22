from scipy.spatial.distance import euclidean

def compare_trees(tree1, tree2, method='node2vec'):
    distance = 0.0
    if method=='node2vec':
      # Convert trees to graphs
      graph1 = process_tree_to_graph(tree1)
      graph2 = process_tree_to_graph(tree2)
  
      # Generate embeddings
      embeddings1 = generate_embeddings(graph1)
      embeddings2 = generate_embeddings(graph2)
  
      # Aggregate embeddings
      tree_embedding1 = aggregate_embeddings(embeddings1)
      tree_embedding2 = aggregate_embeddings(embeddings2)
  
      # Compute Euclidean distance between embeddings
      distance = euclidean(tree_embedding1, tree_embedding2)
    return distance
