import networkx as nx 
from numpy.random import default_rng, Generator
import numpy as np
import pickle

# Generate random graph
def get_watts_strogatz_graph(N,k,p):
    G: nx.Graph = None
    try: 
        with open(f"{N}-{k}-{p}-strogatz.pkl","rb") as f:
            G = pickle.load(f)
    except:
        pass
    if G is None:
        G = nx.generators.connected_watts_strogatz_graph(N,k,p)
        with open(f"{N}-{k}-{p}-strogatz.pkl","wb") as f:
            pickle.dump(G,f)
    return G

# Generate complete graph
def complete_graph(N):
    G: nx.Graph = None
    try: 
        with open(f"{N}-complete.pkl","rb") as f:
            G = pickle.load(f)
    except:
        pass
    if G is None:
        G = nx.generators.complete_graph(N)
        with open(f"{N}-complete.pkl","wb") as f:
            pickle.dump(G,f)
    return G

