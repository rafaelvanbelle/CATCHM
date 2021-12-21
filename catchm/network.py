import networkx as nx

def create_network(edgelist):
    """
    This function creates a the network structure.

    Parameters
    ----------
        edgelist : list
            list of tuples

    Returns:
        G : networkx.Graph
    """
    
    # Create tripartite network with transaction, merchant and cardholder nodes
    G = nx.Graph()

    # Create an ID for each transfer
    transfers = [str(i) for i in range(len(edgelist))]
    senders = [str(i) for i,_ in edgelist]
    receivers = [str(j) for _,j in edgelist]    
    
    G.add_nodes_from(set(receivers), type='receiver')
    G.add_nodes_from(set(senders), type='sender')
    G.add_nodes_from(transfers, type='transfer')

    # Add edges
    G.add_edges_from(zip(transfers, senders))
    G.add_edges_from(zip(transfers, receivers))
    
    return G