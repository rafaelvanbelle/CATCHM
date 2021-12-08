from multiprocessing import Pool
import pandas as pd
from functools import partial
import numpy as np
from tqdm import tqdm

def inductive_pooling(edgelist, embeddings, G, workers, gamma=1000, dict_node=None, average_embedding=True):
    
        #dict of tuples to array
        edgearray = np.array([[str(id), v[0], v[1]] for id,v in enumerate(edgelist)])

        if average_embedding:
            avg_emb = embeddings.mean(axis=0)
        else:
            avg_emb = None
        
        #if __name__ == '__main__':
        result_list = [] 
        with Pool(workers) as p:
            for result in tqdm(p.imap(partial(inductive_pooling_chunk, embeddings=embeddings, G=G, average_embedding=avg_emb), np.array_split(edgearray, workers)), total=len(np.array_split(edgearray, workers))):
                result_list.append(result)
            #r = tqdm(p.map(partial(inductive_pooling_chunk, embeddings=embeddings, G=G, average_embedding=avg_emb), np.array_split(edgearray, workers))

        new_embeddings = np.zeros((len(edgelist), embeddings.shape[1]))
        for result_dict in result_list:
            for id, emb in result_dict.items():
                new_embeddings[int(id), : ] = emb
        
        return new_embeddings
	
def inductive_pooling_chunk(edgearray, embeddings, G, gamma=1000, average_embedding=None):
    
    #Create a container for the new embeddings
    new_embeddings = dict()

    for row in edgearray:

        transfer, sender, receiver = row

        mutual = False    

        if G.has_node(sender) & G.has_node(receiver):
            mutual_neighbors = list(set(G.neighbors(sender)).intersection(set(G.neighbors(receiver))))
            # convert string ids to numerical ids 
            mutual_neighbors = list(map(int, mutual_neighbors))
            # sort numerical ids
            mutual_neighbors.sort()

            
            if (len(mutual_neighbors) > 0): 
                mutual = True
                # take most recent mutual neighbor
                most_recent_mutual_neighbor = mutual_neighbors[-1]
                # Use dataframe with TX_ID on index (to speed up retrieval of transfer rows)
                most_recent_embedding_mutual_neighbor = embeddings[most_recent_mutual_neighbor, :]

                new_embeddings[transfer] = most_recent_embedding_mutual_neighbor
                
                        
        if G.has_node(sender) & (not mutual):

            sender_neighbors = list(map(int, G.neighbors(sender)))
            pooled_embedding = get_pooled_embedding(sender_neighbors, embeddings, gamma)
            
            new_embeddings[transfer] = pooled_embedding
            
        elif G.has_node(receiver) & (not mutual):

            receiver_neighbors = list(map(int, G.neighbors(receiver)))

            pooled_embedding = get_pooled_embedding(receiver_neighbors, embeddings, gamma)

            new_embeddings[transfer] = pooled_embedding
            
            
        elif (not mutual):
            new_embeddings[transfer] = average_embedding
                    

    return new_embeddings
                            
def get_pooled_embedding(neighbors, embeddings, gamma):
    
    embeddings_to_pool = embeddings[neighbors, :]
    most_recent_embeddings_to_pool = embeddings_to_pool[-min(gamma, embeddings_to_pool.shape[0]): , : ]
    
    pooled_embedding = most_recent_embeddings_to_pool.mean(axis=0)
    
    return pooled_embedding