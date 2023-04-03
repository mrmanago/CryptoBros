import networkx as nx
import numpy as np
from numpy.random import default_rng, Generator
from dataclasses import dataclass, replace
import pickle
import datasets
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from model_settings import ModelSettings
import seaborn as sns
from collections import deque
from typing import Iterable

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
# size overrides the steps function and will simulate until graph reaches desired
def simulate(model_settings: ModelSettings, steps: int, seed = None, size = None):
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
    i = 0
    while i < steps or size is not None:
        G, notnat, indices, notnat_ingoing = one_step(G,notnat, indices, notnat_ingoing ,model_settings, rand)
        i += 1
        if G.number_of_nodes() == size:
            break
    return G

def simulate_many_runs(settings: ModelSettings, seed=1234, nr_runs=100, size=1000) -> np.array:
    try:
        with open(f"CryptoNet,{str(seed)},runs={nr_runs},p={str(settings.p)},q={str(settings.q)},nat={str(settings.outgoing_nat)},out={str(settings.outgoing)}.pkl", "rb") as f:
            G = pickle.load(f)
    except:
        G = np.empty(nr_runs, dtype=object)
        for i in range(nr_runs):
            rand = np.random.default_rng(seed + i*0xdeadbeef)
            K = simulate(settings, 1000, seed = seed, size = size)
            G[i] = K
            print("finished iteration " + str(i))
        with open(f"CryptoNet,{str(seed)},runs={nr_runs},p={str(settings.p)},q={str(settings.q)},nat={str(settings.outgoing_nat)},out={str(settings.outgoing)}.pkl", "wb") as f:
            pickle.dump(G, f)
    return G

def round_up(num: float, decimals: int = 0):
    """Round to given number of decimals with positive bias. I could not find it, but is there really no standard function for this?"""
    multiplier = 10 ** decimals
    return math.ceil(num * multiplier) / multiplier

def get_mean_confidence(arr: np.ndarray, name: str, return_string: bool = True) -> str:
    """Given an np.ndarray arr with a name, prints mean and 95%-confidence interval (rounded up generously) for that mean.
    Returns this as a LaTeX-table suited string or an unrounded (mean, halfwidth) pair, depending on return_string=True or return_string=False."""

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    half_width = (std / math.sqrt(len(arr))) * 1.96

    # Compute the significance of the half width, and use it to round the mean
    try:
        significance_of_hw = 1 - math.ceil(math.log10(half_width))
    except ValueError:      # occurs when half_width is zero
        significance_of_hw = 3
    mean_rounded = round(mean, significance_of_hw)

    # Also round the half width itself to one significant digit
    half_width_rounded = round_up(half_width, significance_of_hw)

    # Remove trailing decimal if integral
    if significance_of_hw <= 0:
        mean_rounded, half_width_rounded = int(mean_rounded), int(half_width_rounded)
    
    print(f"\n{name} based on {len(arr)} samples:")
    # print(f"{mean_rounded:.4G} +- {half_width_rounded:.1G}")
    print(f"{mean_rounded} +- {half_width_rounded}")

    # print (f"\n{name} based on {len(arr)} samples (UNROUNDED):")
    # print(f"{mean} +- {half_width}")

    if return_string:
        return f"& {mean_rounded} $\pm$ {half_width_rounded}"
    
    return (mean, half_width)

def plot_degree_variance_for_natting_effects(q: float, outgoing: int, p_range: Iterable, nat_range: Iterable, nr_runs: int = 100,
        seed: int = 1234, size: int = 1000) -> None:
    # Set colors for the different natting probability lines
    COLORS = sns.color_palette(palette='viridis', n_colors=len(p_range))

    df_exit_probs = pd.DataFrame(index=p_range, columns=nat_range)
    df_exit_probs_hws = pd.DataFrame(index=p_range, columns=nat_range)       # half widths of the confidence intervals
    for p in p_range:
        for nat in nat_range:
            # Settings for this run
            settings = ModelSettings(q, p, nat, outgoing)

            # Run simulations and gather results
            # The seed is computed in such a way that every different (q,e,c0)-triple should mostly get a different seed, just in case
            G = simulate_many_runs(settings, seed=int(seed + q + 10000*p + 100*nat), nr_runs=nr_runs, size = size)

            df_results = pd.DataFrame(columns=['variance'])

            for run in range(nr_runs):
                # Extract the results from one run
                degree = np.empty(size,dtype=int)
                i = 0
                for (node, deg) in G[run].degree:
                    degree[i] = deg
                    i += 1
                df_results.loc[run, 'variance'] = np.var(degree)

            print(f"***** q = {q}, p = {p}, out_nat = {nat}, out = {outgoing} *****")

            # Compute degree variance
            df_exit_probs.loc[p, nat], df_exit_probs_hws.loc[p, nat] = get_mean_confidence(df_results['variance'],
                "Degree variance of resulting graph", return_string=False)

            print('\n\n')
    
    print(df_exit_probs)

    plt.figure()

    for p_idx, p in enumerate(p_range):
        plt.errorbar(x=nat_range, y=df_exit_probs.loc[p, :], yerr=df_exit_probs_hws.loc[p, :], capsize=4, elinewidth=1,
            color=COLORS[p_idx], label=f"$p = ${p}", fmt='o--')

    plt.legend()
    plt.xlabel("Outgoing links for natted node", size=14)
    plt.ylabel("Degree variance", size=14)

if __name__ == "__main__":
    #modelsettings = ModelSettings(1,0.5,8,9)
    # test
    #print(simulate_many_runs(modelsettings, nr_runs=10, size = 1000)[0])
    p_range = [0.1,0.2]
    out_range = [1,2]
    plot_degree_variance_for_natting_effects(1,9,p_range,out_range)
    plt.show()