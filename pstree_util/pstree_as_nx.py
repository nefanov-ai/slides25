import networkx as nx

def process_tree_to_graph(tree):
    """
    Convert a process tree to a directed graph.
    """
    graph = nx.DiGraph()
    for pid, process in tree.items():
        graph.add_node(pid, comm=process["comm"])
        if process["ppid"] != 0:  # Skip the root process
            graph.add_edge(process["ppid"], pid)
    return graph
