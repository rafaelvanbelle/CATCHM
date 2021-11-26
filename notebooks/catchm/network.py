import networkx as nx
import numpy as np

def create_network(edgelist):
    """This function creates a the network structure.

    Args:
        edgedict (dict): keys contain transaction ids, values contain tuples with edges
    """

    # Create tripartite network with transaction, merchant and cardholder nodes
    G = nx.Graph()

    # Create an ID for each transfer
    transfers = [str(i) for i in range(len(edgelist))]
    senders = [i for i,_ in edgelist]
    receivers = [j for _,j in edgelist]    
    
    G.add_nodes_from(set(receivers), type='receiver')
    G.add_nodes_from(set(senders), type='sender')
    G.add_nodes_from(transfers, type='transfer')

    # Add edges
    G.add_edges_from(zip(transfers, senders))
    G.add_edges_from(zip(transfers, receivers))
    
    return G