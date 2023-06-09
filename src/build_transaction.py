import pandas as pd
import os
from utils import scores2coalition
import torch
from ego_graph_dataset import select_active_graph
import numpy as np
from tqdm import tqdm


def build_counting_transaction(path, metric, rule, dataset_name, graph_ids=[], fixed_size=False, size=None, sparsity=0.5,
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
        graph_feature = data[graph_id].x.reshape(-1).tolist()
        node_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                     11: "K", 12: "Li", 13: "Ca"}
        nodes_score = df[metric].values
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
        important_nodes_index = top_idx[:cutoff]
        # Add neighbors of the coalition
        if with_neighbors:
            for node in important_nodes_index:
                important_nodes_index.extend(data[graph_id].edge_index[1][data[graph_id].edge_index[0] == node].tolist())
        important_nodes_index = list(set(important_nodes_index))
        # Get the labels of nodes in important_nodes_index
        important_node_label = graph_feature[important_nodes_index]
        # Get the labels of nodes in the graph

        label_count = {
            'C': 0, 'O': 0, 'Cl': 0, 'H': 0, 'N': 0, 'F': 0, 'Br': 0, 'S': 0, 'P': 0, 'I': 0,
            'Na': 0, 'K': 0, 'Li': 0,
            'Ca': 0}
        for label in important_node_label:
            label_count[node_dict[label]] += 1

        dict_line = [f'{label}_{i}' for label, count in label_count.items() for i in range(1, 6) if count >= i]
        result_series.loc[graph_id] = dict_line
    if len(skipped_index) > 0:
        # Drop skipped index in the result series
        result_series.drop(labels=[skipped[0] for skipped in skipped_index], inplace=True)
        print(
            f'Skipped {len(skipped_index)} graphs because the corresponding file is empty. The size of the graph can lead to a to long HN value computation and therefore, we chose to skip the graph. Here is the detail of skipped graphs with their size:\n {skipped_index}')
        print([skipped[0] for skipped in skipped_index])
        # Save transaction
    result_series.to_csv(os.path.join(path, f"transaction_count_{dataset_name}_{rule}_rule.csv"))
    return result_series
metric = 'entropy'
rule= 23
nb_graphs = 4300
method = 'split_top'
transactions = build_counting_transaction('results/mutagenicity/gcn/gstarx', metric=metric, rule=rule,
                  dataset_name="mutagenicity", graph_ids=np.arange(0,3400), fixed_size=True, size=3, sparsity=0.5, method=method)

#
#
# def build_counting_transaction(path, metric, rule, dataset_name, graph_ids=[], fixed_size=False, size=None,
#                                sparsity=0.5, method='split_top', with_neighbors=False):
#     r"""
#     Build a transaction file with the counting label, used for LCM algorithm
#     :param path: path to the folder containing the nodes score
#     :param metric: taquet metric
#     :param rule: targeted rule
#     :param dataset_name: name of the dataset
#     :param graph_ids: list of IDs of graphs to consider
#     :param fixed_size: boolean to indicate if we want to fix the size of the coalition or using sparisity parameter
#     :param size: size of the coalition if fixed_size is True
#     :param sparsity: sparsity of the coalition if fixed_size is False
#     :param method: method to select the coalition
#     :param with_neighbors: boolean to indicate if we want to consider the neighbors of the coalition
#     :return: pandas Series transactions
#     """
#     labels = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
#     counting_label = [f'{label}{i}' for label in labels for i in range(6)]
#     # Initialize the result pandas series
#     result_series = pd.Series(
#         index=graph_ids)
#     # Load graphs from file
#     data = select_active_graph(
#         f"./activ_ego/mutag_{rule}labels_egos.txt",
#         index_to_select=graph_ids)
#     skipped_index = []
#     for graph_id in tqdm(graph_ids):
#         df = pd.read_csv(os.path.join(path, f"rule_{rule}/result_{dataset_name}_{rule}_{graph_id}.csv"))
#         graph_feature = data[graph_id].x.reshape(-1).tolist()
#         node_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
#                      11: "K", 12: "Li", 13: "Ca"}
#         inv_node_dict = {v: k for k, v in node_dict.items()}
#         nodes_score = df[metric].values
#         if nodes_score is None or len(nodes_score) == 0:
#             skipped_index += [(graph_id, data[graph_id].x.shape[0])]
#             continue
#         node_label = [node_dict[i] for i in graph_feature]
#         scores_tensor = torch.tensor(nodes_score)
#         top_idx = scores_tensor.argsort(descending=True)
#         sorted_label = [node_label[i] for i in top_idx]
#         if method == 'fixed_size':
#             assert size is not None
#             cutoff = size
#         elif method == 'split_top':
#             top = np.array(nodes_score)
#             top = np.sort(top)[::-1]
#             split_index = 1
#             for i in range(1, len(top) - 1):
#                 if top[i] < 0:
#                     split_index = i
#                     break
#             cutoff = split_index
#         else:
#             cutoff = int(len(nodes_score) * (1 - sparsity))
#             cutoff = min(cutoff, (scores_tensor > 0).sum().item())
#         coalition = top_idx[:cutoff]
#         # Create a list of nodes in the coalition and direct neighbors
#         if with_neighbors:
#             for node in coalition:
#                 neighbors = data[graph_id].edge_index[1][data[graph_id].edge_index[0] == node]
#                 coalition = np.concatenate((coalition, neighbors))
#         coalition_label = []
#         for node in coalition:
#             coalition_label.append(node_dict[node_label[node]])
#         # Count occurence of each element in coalition
#         label_count = {
#             'C': 0, 'O': 0, 'Cl': 0, 'H': 0, 'N': 0, 'F': 0, 'Br': 0, 'S': 0, 'P': 0, 'I': 0,
#             'Na': 0, 'K': 0, 'Li': 0,
#             'Ca': 0}
#         dict_line = [f'{label}_{i}' for label, count in label_count.items() for i in range(1, 6) if count >= i]
#         result_series.loc[graph_id] = dict_line
#     if len(skipped_index) > 0:
#         # Drop skipped index in the result series
#         result_series.drop(labels=[skipped[0] for skipped in skipped_index], inplace=True)
#         print(
#             f'Skipped {len(skipped_index)} graphs because the corresponding file is empty. The size of the graph can lead to a to long HN value computation and therefore, we chose to skip the graph. Here is the detail of skipped graphs with their size:\n {skipped_index}')
#         print([skipped[0] for skipped in skipped_index])
#     # Save transaction
#     result_series.to_csv(os.path.join(path, f"transaction_count_{dataset_name}_{rule}_rule.csv"))
#     return result_series
