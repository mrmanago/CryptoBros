import networkx as nx
import numpy as np
from numpy.random import default_rng, Generator
from dataclasses import dataclass, replace
import pickle
import datasets
from model_settings import ModelSettings
from collections import deque

# Perform one step
# G: the current network
# notnat: indices of nodes that are not natted
# indices: indices of removed nodes
# notnat_ingoing: all the nodes that established a connection with a particular notnatted node
# model_settings: the settings that can be used
def one_step(G: nx.Graph, notnat: set, indices: deque, notnat_ingoing: dict, model_settings: ModelSettings, rand: Generator) -> nx.Graph:
    if rand.random() < model_settings.q:  # Add node
        if len(indices) == 0: # index to use for next node
            nextnode = G.size()
        else:
            nextnode = indices.pop()
        G.add_node(nextnode)
        if rand.random() < model_settings.p: # Natted
            newout = rand.choice(np.array(list(notnat)), size=model_settings.outgoing_nat, replace=False)
        else: # not natted
            newout = rand.choice(np.array(list(notnat)), size=model_settings.outgoing, replace=False)
            notnat.add(nextnode)
            notnat_ingoing[nextnode] = []
        for l in newout:
            notnat_ingoing[l].append(nextnode)
            G.add_edge(nextnode, l)
    else: # Exit node
        selected_node = rand.integers(G.number_of_nodes())
        indices.append(selected_node)
        if selected_node in notnat: # rewire outgoing connections
            notnat.remove(selected_node)
            for i in notnat_ingoing[selected_node]:
                alreadylinked = set(G.adj[i].keys())
                viable = notnat.intersection(alreadylinked)
                if len(viable) > 0:
                    newcon = rand.choice(np.array(list(viable)))
                    G.add_edge(i,newcon)
        G.remove_node(selected_node) # remove node
    return G

### TODO: 
# - ensure no indexing issues when sampling from a small notnat
# - add proper handling of when there are not viable candidates to rewire too (in exit strategy)

if __name__ == "__main__":
    G = datasets.complete_graph(10)
    modelsettings = ModelSettings(0.5,0.5,8,9)
    rand = default_rng()
    notnat_ingoing = {}
    for i in range(0,10):
        notnat_ingoing[i] = [j for j in range(0,10) if j != i]
    one_step(G, set(range(0,10)), deque(), notnat_ingoing ,modelsettings,rand)