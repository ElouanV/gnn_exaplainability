import torch
import networkx as nx
import numpy as np
import os
from pattern_mining import compare_structure_and_feature
import utils
from dataset.dataset_loaders import load_dataset
from scipy.special import softmax
from torch_geometric.utils import to_dense_adj

number_of_graph_per_rule = utils.MUTAGENICITY_NUMBER_OF_GRAPH_PER_RULE
mutag_label_dict = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                    11: "K", 12: "Li", 13: "Ca"}


def to_torch_graph(graphs, task):
    """
    Transforms the numpy graphs to torch tensors depending on the task of the model that we want to explain
    :param graphs: list of single numpy graph
    :param task: either 'node' or 'graph'
    :return: torch tensor
    """
    if task == 'graph':
        return [torch.tensor(g) for g in graphs]
    else:
        return torch.tensor(graphs)


def get_graph_indices_from_a_rule(rule, dataset, path_to_active_ego):
    graph_index = []
    file_path = os.path.join(path_to_active_ego, f'mutag_{rule}labels_egos.txt')
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('t') and line.split(' ')[2] != '-1':
                # t # 2526# 0 4332 32
                tokens = line.replace(',', '').replace('(', '').replace(')', '').split(' ')
                cls, g_id, center_node = int(tokens[3]), int(tokens[4]), int(tokens[5])
                graph_index.append((cls, g_id, center_node))
    return graph_index


def parse_gspan_result(path, label_dict):
    """
    Parse the result of gSpan and return the list of graphs
    :param path: the path to the result file
    :param label_dict: the label dictionary
    :return: the list of graphs
    """
    graphs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('t'):  # t # {graph_id} {support}
                graph = nx.Graph()
                support = float(line.split(' ')[3])
                where = line.replace('[', '').replace(']', '').replace(',', '').split(' ')[4]
                where = [int(x) for x in where]
                graphs.append((graph, support, where))
            elif line.startswith('v'):
                _, node_id, label = line.strip().split(' ')
                graph.add_node(int(node_id), label=label_dict[int(label)])
            elif line.startswith('e'):
                _, node_id1, node_id2, label = line.strip().split(' ')
                graph.add_edge(int(node_id1), int(node_id2))
    return graphs


def load_motifs(dataset):
    names = {"ba2": ("ba2"),
             "aids": ("Aids"),
             "BBBP": ("Bbbp"),
             "mutagenicity": ("Mutag"),
             "DD": ("DD"),
             "PROTEINS_full": ("Proteins")}
    name = names[dataset]
    file = "./datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            label = int(l.split(" ")[3].split(":")[1])
            rules.append((label, r, l))
    return rules


def apply_structure_mask(dataset, rule, path_to_active_ego, model, node_selection='split_top',
                         metric='entropy', with_neighbors=True):
    rule_graph_info = get_graph_indices_from_a_rule(rule, dataset, path_to_active_ego)
    gspan_path = f'./results/{dataset}_{rule}_{metric}_split_top_with_neighbors.txt'
    gspan_result = parse_gspan_result(gspan_path, mutag_label_dict)

    ratio_support = compare_structure_and_feature(rule, metric, node_selection, with_neighbors=with_neighbors,
                                                  min_supp_ratio=0.7,
                                                  dataset='mutagenicity', verbose=False)
    if ratio_support is None or len(ratio_support) == 0:
        print('No rule found')
        return None
    explanation = ratio_support[0]
    graphs, features, labels, _, _, test_mask = load_dataset("mutag")
    task = "graph"
    graphs = to_torch_graph(graphs, task)
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    motif = load_motifs("mutagenicity")[rule]
    radius_max = int(motif[1][2]) + 1
    target_class = int(motif[0])
    fidelities = []
    for (cls, g_id, center_node) in rule_graph_info:
        # Get prediction for the graph with the model
        feature = features[g_id].detach()
        g = graphs[g_id].detach()

        # Remove self-loops
        g = g[:, (g[0] != g[1])]

        with torch.no_grad():
            base_score = softmax(model(feature, g))
        print(f'Base score: {base_score[target_class]}')

        perturbed_g = g
        perturbed_feature = feature
        if explanation[1] != -1:
            print(f'Structural explanation: {explanation[0]}')
            pass
        else:
            print(f'Feature explanation: {explanation[0]}')
            # Removes the features extracted
            feature_count = explanation[3].copy()
            # Remove n times each feature accordinf to feature_count dict in the graph starting from the center node
            # with a BFS
            nodes_to_remove = []
            queue = [center_node, -1]
            radius = 0
            while True:
                if not queue:
                    break
                current_node = queue.pop(0)
                if current_node == -1 and queue:
                    radius += 1
                    queue.append(-1)
                    continue
                if radius > radius_max:
                    break
                # Get feature of the current node
                node_feature = feature[current_node].detach().numpy()
                current_feature = node_feature.argmax()
                if feature_count[mutag_label_dict[current_feature]] > 0:
                    nodes_to_remove.append(current_node)
                    perturbed_feature[current_node] = torch.tensor([0 for _ in range(len(feature[current_node]))])
                    feature_count[mutag_label_dict[current_feature]] -= 1
                    # If all vlaues in feature_count are 0, we can stop
                    if all([x == 0 for x in feature_count.values()]):
                        break
                # Get neighbors of the current node using the graph which is a edge_index
                neighbors = g[:, g[0] == current_node][1].detach().numpy()
                queue.extend(neighbors)


            # Remove the nodes from the graph
            perturbed_g = g
            for node in nodes_to_remove:
                perturbed_g = perturbed_g[:, (perturbed_g[0] != node) & (perturbed_g[1] != node)]
        with torch.no_grad():
            perturbed_score = softmax(model(perturbed_feature, perturbed_g))
        print(f'Perturbed score: {perturbed_score[target_class]}')
        fidelity = base_score[target_class] - perturbed_score[target_class]
        fidelities.append(fidelity)
        print(f'Fidelity: {fidelity}')
        print()
        print("--------------------------------------------------")
    print()
    print(f'Mean fidelity: {np.mean(fidelities)}')

from model.model_selector import model_selector

model, checkpoint = model_selector("GNN", "mutagenicity", pretrained=True, return_checkpoint=True)
apply_structure_mask("mutagenicity", 1, path_to_active_ego='./activ_ego/', model=model, node_selection='split_top', )
