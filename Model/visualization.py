import numpy as np
import pickle
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
import pygraphviz as pgv
import seaborn as sns
import networkx as nx
from dataclasses import replace
from model_settings import ModelSettings
from collections import defaultdict
import model
import datasets

# Choose colors for vizualization
opinion_colors = ["#1f78b4","#dc143c"]



# Use this function along with a result generated from 'run_visualization_simulation'
def make_animation(graph_name: str, evolution):

    # Depends on size of graph, for now, i just hardcoded it
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    fig.tight_layout()
    #colors, blue, red = determine_colors_for_nodes(result.initial_ops)
    #node_count = {opinion_colors[0]: blue, opinion_colors[1]: red}
    # For animate function to know stepsizes
    pos = {}

    def init():
        nonlocal pos
        ax.cla()
        G, index = evolution[0]
        colors = determine_colors_for_nodes(G, index)
        pos = default_layout(G,{})
        nx.draw_networkx(G, pos=pos, node_color=colors, with_labels=False, node_size=100, ax=ax)
        #plt.legend(handles=node_count_legend(node_count[opinion_colors[0]],node_count[opinion_colors[1]]))


    def animate(i):
        nonlocal pos
        ax.cla()
        G, index = evolution[i]
        colors = determine_colors_for_nodes(G, index)
        pos = default_layout(G,{})
        nx.draw_networkx(G, pos=pos, node_color=colors, with_labels=False, node_size=100,ax=ax)
        #plt.legend(handles=node_count_legend(node_count[opinion_colors[0]],node_count[opinion_colors[1]]))
        

    anim = FuncAnimation(fig, animate,init_func=init,
        frames = len(evolution), interval = 1)
    
    FFwriter = FFMpegWriter(fps=5)
    anim.save(f'name={graph_name}.mp4', writer = FFwriter)



def determine_positions_from_dot_string(s: str):
    """Given a precomputed layout in dot format determines the positions of each node in from of a dict. 
    """
    s = s.split('{')
    s = s[1].split(';')
    positions_of_nodes = {}
    for string in s:
        string = string.replace('\n','')
        string = string.replace('\t','')
        string = string.split(' ')
        if string[0][0].isdigit() and len(string) == 1:
            index = re.search(r'\d+', string[0]).group()
            posx = re.search(r'pos=',string[0]) 
            posx = string[0][posx.span()[1]:].split(',')
            posy = float(posx[1].replace('"',''))
            posx = float(posx[0].replace('"',''))
            positions_of_nodes[int(index)] = (posx,posy)
    return positions_of_nodes

# Make color array based on opinions array
def determine_colors_for_nodes(G: nx.graph, index):
    colors = []
    for node in G.nodes:
        if node == index:
            colors.append(opinion_colors[0])
        else:
            colors.append(opinion_colors[1])
    return (colors)

def default_layout(G: nx.Graph, pos: dict[int,tuple]) -> dict[int,tuple]:
    B = nx.nx_agraph.to_agraph(G)
    B.graph_attr.update(normalize="true")
    for index in pos:
        posx,posy = pos[index]
        B.get_node(str(index)).attr["pin"] = "true"
        B.get_node(str(index)).attr["pos"] = str(posx) + "," + str(posy)
    B.layout(prog='fdp')
    s = B.to_string()
    return determine_positions_from_dot_string(s)

def lattice_layout(n,m,G):
    # fit lattice in (0,0) to (2000,2000)
    stepx = 2000 / n
    stepy = 2000 / m
    pos = {}
    posx = stepx
    posy = stepy
    for i in range(n):
        for j in range(m):
            pos.update({20*i + j: (posx, posy)})
            posx += stepx
        posx = stepx
        posy += stepy
    return pos


def node_count_legend(blue,red):
    return [mpl.lines.Line2D([0], [0],marker='o', color='w', label=f'{blue}',markerfacecolor=opinion_colors[0], markersize=10),
        mpl.lines.Line2D([0], [0], marker='o', color='w', label=f'{red}',markerfacecolor=opinion_colors[1], markersize=10)]   



def test_fig_layout(G: nx.graph, pos: dict[int,tuple], index):
    # Depends on size of graph, for now, i just hardcoded it
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    colors = determine_colors_for_nodes(G, index)
    nx.draw_networkx(G, pos=pos, node_color=colors, with_labels=False, node_size=100,ax=ax)
    fig.tight_layout()
    plt.show()

def main():
    # Example of how to make viz
    modelsettings = ModelSettings(1,0.7,1,8)
    # test
    evolution = model.getEvolution(modelsettings, 1000)

    # name graph
    name = 'basic'
    G, index = evolution[1]
    pos = default_layout(G,{})
    G, index = evolution[2]
    pos = default_layout(G,pos)
    G, index = evolution[3]
    pos = default_layout(G,pos)

    # Run simulations and gather results -> use this function
    # if you would like to use default, u can leave out specifying parameter
    #result, pos = run_vizualization_simulation(settings, run_until_absorbtion=run_absorb, epochs=epochs,initial_c=initial_c, G = G, graph_name=name, layoutalgo=lambda G: lattice_layout(20,20,G))

    print(evolution[:100])
    make_animation(name,evolution[:100])
    #test_fig_layout(G,pos, index)

    # Experimentation using pygraphviz
    #A = nx.nx_agraph.to_agraph(result.final_G)
    #A.graph_attr['overlap_scaling'] = '30'
    #A.graph_attr['K'] = '0.8'
    #A.layout(prog='fdp')
    #A.draw('test.png')

if __name__ == "__main__":
    main()