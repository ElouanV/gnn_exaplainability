import pandas as pd
import os
from utils import scores2coalition
import torch
from ego_graph_dataset import select_active_graph
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


def build_counting_transaction(path, metric, rule, dataset_name, graph_ids=[], fixed_size=False, size=None,
                               sparsity=0.5,
                               method='split_top', with_neighbors=True):
    r"""
    Build a transaction file with the coalition label, used for pattern mining. Transactions are built from the
    scores of the nodes of the graph. The nodes are sorted by their score and the top nodes are selected to form
    the coalition. Transaction of a graph is the list of labels that are present in the graph but not in the
    coalition.
    :param path:
    :param metric:
    :param rule:
    :param dataset_name:
    :param graph_ids:
    :param fixed_size:
    :param size:
    :param sparsity:
    :param method:
    :param with_neighbors:
    :return: pandas DataFrame transaction
    """
    result_series = pd.Series(
        index=graph_ids)
    # Load graphs from file
    data = select_active_graph(
        f"./activ_ego/mutag_{rule}labels_egos.txt",
        index_to_select=graph_ids)
    skipped_index = []
    for graph_id in tqdm(graph_ids):
        df = pd.read_csv(os.path.join(path, f"rule_{rule}/result_{dataset_name}_{rule}_{graph_id}.csv"))
        graph = to_networkx(data[graph_id], to_undirected=True, node_attrs=['center', 'x'])
        node_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                     11: "K", 12: "Li", 13: "Ca"}
        nodes_score = df[metric].values
        if nodes_score is None or len(nodes_score) == 0:
            skipped_index.append((graph_id, 'no score'))
            continue
        scores_tensor = torch.tensor(nodes_score)
        top_idx = scores_tensor.argsort(descending=True).tolist()
        if method == 'split_top':
            top = np.array(nodes_score)
            top = np.sort(top)[::-1]
            split_index = 1
            for i in range(1, len(top) - 1):
                if top[i] < 0:
                    split_index = i
                    break
            cutoff = split_index
        elif method == 'fixed_size':

            assert size is not None
            cutoff = size
        else:
            cutoff = int(len(nodes_score) * (1 - sparsity))
            cutoff = min(cutoff, (scores_tensor > 0).sum().item())
        best_scores_nodes = top_idx[:cutoff]
        # Add neighbors of the coalition
        for node in graph.nodes():
            if graph.nodes[node]['center'] and node not in best_scores_nodes:
                best_scores_nodes.append(node)

        important_nodes_index = best_scores_nodes.copy()

        if with_neighbors:
            for node in best_scores_nodes:
                important_nodes_index.extend(
                    data[graph_id].edge_index[1][data[graph_id].edge_index[0] == node].tolist())
        important_nodes_index = set(important_nodes_index)
        # Convert important_nodes_index to slices
        important_nodes_index = list(important_nodes_index)
        # Get the labels of nodes in important_nodes_index
        important_node_label = [node_dict[data[graph_id].x[important_node_index].item()] for important_node_index in
                                important_nodes_index]
        # Get the labels of nodes in the graph

        label_count = {
            'C': 0, 'O': 0, 'Cl': 0, 'H': 0, 'N': 0, 'F': 0, 'Br': 0, 'S': 0, 'P': 0, 'I': 0,
            'Na': 0, 'K': 0, 'Li': 0,
            'Ca': 0}
        for label in important_node_label:
            label_count[label] += 1

        dict_line = [f'{label}_{i}' for label, count in label_count.items() for i in range(1, 6) if count >= i]
        result_series.loc[graph_id] = dict_line
    if len(skipped_index) > 0:
        # Drop skipped index in the result series
        result_series.drop(labels=[skipped[0] for skipped in skipped_index], inplace=True)
        # Save transaction
    result_series.to_csv(os.path.join(path, f"transaction_count_{dataset_name}_{rule}_rule.csv"))
    return result_series
