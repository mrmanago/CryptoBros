import networkx as nx
import numpy as np
from numpy.random import default_rng, Generator
from dataclasses import dataclass, replace
import pickle
import datasets
from model_settings import ModelSettings
from collections import deque

def clean_ingoing(notnat_ingoing: dict, selected_node: int):
    for key in notnat_ingoing:
        notnat_ingoing[key].discard(selected_node)
    return notnat_ingoing



# Perform one step
# G: the current network
# notnat: indices of nodes that are not natted
# indices: indices of removed nodes
# notnat_ingoing: all the nodes that established a connection with a particular notnatted node
# model_settings: the settings that can be used
def one_step(G: nx.Graph, notnat: set, indices: deque, notnat_ingoing: dict, model_settings: ModelSettings, rand: Generator) -> (nx.Graph, set, deque, dict):
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
            notnat_ingoing[nextnode] = set()
        for l in newout:
            notnat_ingoing[l].add(nextnode)
            G.add_edge(nextnode, l)
    else: # Exit node
        selected_node = rand.choice(np.array(G.nodes))
        indices.append(selected_node)
        if selected_node in notnat: # rewire outgoing connections
            notnat.remove(selected_node)
            for i in notnat_ingoing[selected_node]:
                if not G.has_node(i):
                    print(f"Does not have node {i}")
                    raise ValueError()
                alreadylinked = set(G.adj[i].keys())
                viable = notnat.intersection(alreadylinked)
                if len(viable) > 0:
                    newcon = rand.choice(np.array(list(viable)))
                    G.add_edge(i,newcon)
            notnat_ingoing.pop(selected_node)
        notnat_ingoing = clean_ingoing(notnat_ingoing, selected_node) # Ensures on occurences of removed node
        G.remove_node(selected_node) # remove node
    return G, notnat, indices, notnat_ingoing

### TODO: 
# - ensure no indexing issues when sampling from a small notnat
# - add proper handling of when there are not viable candidates to rewire too (in exit strategy)

# Simulate the evolution of the graph for {steps} runs
# model_settings: specify parameters for simulation
def simulate(model_settings: ModelSettings, steps: int, seed = None):
    rand = default_rng(12345)
    if seed is not None:
        rand = default_rng(seed)
    N = model_settings.outgoing + 1
    G = datasets.complete_graph(N)
    notnat_ingoing = {}
    for i in range(N):
        notnat_ingoing[i] = set([j for j in range(N) if j != i])
    notnat = set(range(N))
    indices = deque()
    for i in range(steps):
        G, notnat, indices, notnat_ingoing = one_step(G,notnat, indices, notnat_ingoing ,modelsettings, rand)
    return G


if __name__ == "__main__":
    modelsettings = ModelSettings(0.78,0.5,8,9)
    print(simulate(modelsettings, 1000))