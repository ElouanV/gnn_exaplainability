import torch
from torch_geometric.data import Data, InMemoryDataset
from parse_active import build_graphs_from_file
import networkx as nx


def select_active_graph(filepath, num_class=2, index_to_select: list = []):
    """
    Select the active graph from the mutagenicity dataset and return the corresponding PyTorch Geometric Data object
    :param filepath: the path to the mutagenicity dataset
    :param num_class: the number of classes in the dataset
    index_to_select: the index of the graphs to select, if empty, select all graphs
    :return: the list of PyTorch Geometric Data object

    remark: here we only support the mutagenicity dataset, add your own label dict if needed
    """
    # Mutagenicity label dict
    label_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                  11: "K", 12: "Li", 13: "Ca"}
    inv_label_dict = {v: k for k, v in label_dict.items()}
    graphs, _ = build_graphs_from_file(filepath, num_class)
    graphs = graphs[0] + graphs[1]
    print(f"Number of graphs: {len(graphs)}")
    if not index_to_select:
        index_to_select = list(range(len(graphs)))
    selected_graph = [graphs[i] for i in index_to_select]
    return [convert_active_graph_to_data(graph, inv_label_dict) for graph in selected_graph]


def convert_active_graph_to_data(graph, label_dict):
    """
    Convert a networkx graph to a PyTorch Geometric Data object, this is not equivalent to 'from_networkx' function of
    PyG because here we renumber the nodes from 0 to n and reorganize the edge_index
    :param graph: the networkx graph
    :param label_dict: the label dict
    :return: the PyTorch Geometric Data object
    """
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
    """
    The dataset class for the ego graph dataset extracted from activation rule mining as PyG dataset
    """
    def __init__(self, path, index_list):
        """
        :param path: the path to the ego graph dataset
        :param index_list: the list of index of the graphs to select, if empty, select all graphs
        """
        super().__init__()
        self.dataset_root = path
        self.index_list = index_list
        self.data = select_active_graph(path, index_to_select=self.index_list)
        if not self.index_list:
            self.index_list = list(range(len(self.data)))
        self.index_dic = {k: i for i, k in enumerate(self.index_list)}

    def __len__(self):
        """
        :return: the number of graphs in the dataset
        """
        return len(self.index_list)

    def __getitem__(self, index):
        """
        :param index: the index of the graph to get
        :return: the PyTorch Geometric Data object
        """
        return self.data[self.index_dic[index]]

    def get_index_list(self):
        """
        :return: the list of index of the graphs in the dataset
        """
        return self.index_list
