import torch
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from parse_active import build_graphs_from_file
import networkx as nx


def select_active_graph(filepath, num_class=2, target=0, index_to_select=[]):
    # Mutagenicity label dict
    label_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                  11: "K", 12: "Li", 13: "Ca"}
    inv_label_dict = {v: k for k, v in label_dict.items()}
    graphs, _ = build_graphs_from_file(filepath, num_class)
    graphs = graphs[0] + graphs[1]
    print(f"Number of graphs: {len(graphs)}")
    if index_to_select == []:
        index_to_select = list(range(len(graphs)))
    selected_graph = [graphs[i] for i in index_to_select]
    return [convert_active_graph_to_data(graph, inv_label_dict) for graph in selected_graph]


def convert_active_graph_to_data(graph, label_dict):
    # Renumbering node from 0 to n
    node_dict = {k: i for i, k in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, node_dict)

    feature_dic = nx.get_node_attributes(graph, name="label")
    center_dic = nx.get_node_attributes(graph, name="center")
    edge_index = [[], []]
    for src, dest in graph.edges():
        edge_index[0].append(src)
        edge_index[1].append(dest)

    node_index = {k: i for i, k in enumerate(feature_dic.keys())}
    # Inverse the order if v > u
    edge_index = [[min(src, dest) for src, dest in zip(edge_index[0], edge_index[1])],
                  [max(src, dest) for src, dest in zip(edge_index[0], edge_index[1])]]
    feature_matrix = {node_index[k]: v for k, v in feature_dic.items()}
    # Sort feature_matrix on key
    feature_matrix = {k: feature_matrix[k] for k in sorted(feature_matrix.keys())}
    center_dic = {node_index[k]: v for k, v in center_dic.items()}
    # Sort center_dic on key
    center_dic = {k: center_dic[k] for k in sorted(center_dic.keys())}
    center_list = [center_dic[k] for k in sorted(center_dic.keys())]
    center = torch.tensor(center_list).reshape(-1, 1)
    # Get a list of values from the dictionary
    feature_matrix = [label_dict[v] for v in feature_matrix.values()]
    feature_matrix = torch.tensor(feature_matrix).reshape(-1, 1)
    # Create a Data object from the graph
    data = Data(x=feature_matrix, edge_index=torch.tensor(edge_index, dtype=torch.long), center=center)
    return data


class EgoGraphDataset(InMemoryDataset):
    def __init__(self, path, index_list):
        self.dataset_root = path
        self.index_list = index_list
        self.data = select_active_graph(path, index_to_select=self.index_list)
        if self.index_list == []:
            self.index_list = list(range(len(self.data)))
        self.index_dic = {k: i for i, k in enumerate(self.index_list)}

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        target = self.index_dic[index]
        return self.data[self.index_dic[index]]

    def get_index_list(self):
        return self.index_list
