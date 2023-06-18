
import numpy as np
from math import log
import utils
from ego_graph_dataset import select_active_graph
from tqdm import tqdm
import pandas as pd
import os
from utils import scores2coalition, to_networkx, get_feature_dict
from utils import to_networkx

from gspan_mine.gspan_mining.config import parser
from gspan_mine.gspan_mining.main import main
import networkx as nx
from skmine.itemsets import LCM
import matplotlib.pyplot as plt

number_of_graph_per_rule = utils.MUTAGENICITY_NUMBER_OF_GRAPH_PER_RULE


def pattern_mining(transactions, nb_graphs, supp_ratio=0.5):
    """
    Mine closed itemsets from transactions using LCM
    :param transactions: the transactions to mine
    :param nb_graphs: the number of graphs in the dataset
    :param supp_ratio: the minimum support ratio
    :return: the pattern mined
    """
    min_supp = int(nb_graphs * supp_ratio)
    lcm = LCM(min_supp=min_supp, n_jobs=4)
    pattern = lcm.fit_transform(transactions)
    return pattern


def build_subgraph_and_transactions(dataset, rule, node_selection, nb_graph, metric, with_neighbors, data_path):
    """
    Build subgraphs and transactional data from the active ego graphs to be used in gSpan and LCM. Graphs data are
    saved in data_path with the format required by gSpan. Transactional data are return with the format
    required by LCM. Both can be directly used by respective algorithms without any further processing.
    :param dataset: the dataset to use
    :param rule: the rule to use
    :param node_selection: the node selection method to use
    :param nb_graph: the number of graphs in the dataset
    :param metric: the metric to use to select the nodes
    :param with_neighbors: whether to add the neighbors of the nodes in the coalition to the coalition
    :param data_path: the path to save the graphs data
    :return: the transactional data
    """
    graphs = select_active_graph(f'./activ_ego/mutag_{rule}labels_egos.txt', 2, [])
    skipped_index = []
    feature_dict = get_feature_dict(dataset)
    transactions = pd.Series(dtype=object,
                             index=np.arange(0, nb_graph))
    with open(data_path, 'w+') as f:
        for graph_id, graph_data in tqdm(enumerate(graphs)):
            df_node_scores = pd.read_csv(os.path.join("./results/mutagenicity/gcn/gstarx",
                                                      f"rule_{rule}/result_{dataset}_{rule}_{graph_id}.csv"))
            graph = to_networkx(graph_data, to_undirected=True, node_attrs=['center', 'x'])
            if df_node_scores is None or len(df_node_scores) == 0:
                skipped_index.append(graph_id)
                transactions[graph_id] = []
                continue
            node_score = df_node_scores[metric].values
            coalition = scores2coalition(node_score, sparsity=0.5, fixed_size=True, size=3, method=node_selection)
            # If the node that has the label 'center' to True in the graph is not in the coalition, add it
            for node in graph.nodes():
                if graph.nodes[node]['center'] and node not in coalition:
                    coalition.append(node)
            # Add to the colaition the neighbors of the nodes in the coalition
            if with_neighbors:
                for node in coalition:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in coalition:
                            coalition.append(neighbor)
            # Build _transactions
            subgraph = graph.subgraph(coalition)
            label_count = {
                'C': 0, 'O': 0, 'Cl': 0, 'H': 0, 'N': 0, 'F': 0, 'Br': 0, 'S': 0, 'P': 0, 'I': 0,
                'Na': 0, 'K': 0, 'Li': 0,
                'Ca': 0}
            for node in subgraph.nodes():
                label_count[feature_dict[subgraph.nodes[node]['x']]] += 1
            dict_line = [f'{label}_{i}' for label, count in label_count.items() for i in range(1, 6) if count >= i]
            transactions.loc[graph_id] = dict_line

            # Build subgraph and write it as .data file
            f.write(f't # {graph_id}\n')
            for node in subgraph.nodes():
                f.write(f'v {int(node)} {graph.nodes[node]["x"]}\n')
            for edge in subgraph.edges():
                f.write(f'e {int(edge[0])} {int(edge[1])} 0\n')
    return transactions


def gspan_mine_rule(nb_graph, supp_ratio=0.9, save_path='./results', data_path=''):
    """
    Mine patterns from the graphs in data_path using gSpan
    :param nb_graph: the number of graphs in the dataset
    :param supp_ratio: the minimum support ratio
    :param save_path: the path to save the patterns
    :param data_path: the path to the graphs data
    :return: the patterns mined
    :remark: the patterns are saved in save_path, the gSpan module used here is a bit modified to save the patterns as
    we need
    """
    min_support = int(nb_graph * supp_ratio)
    args_str = f'-s {min_support} -p False -d False --save-path {save_path} {data_path}'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = main(FLAGS)
    return gs


def lcm_pattern_to_count(patterns: pd.DataFrame, label_dict: dict):
    """
    Transform the patterns mined by LCM to a list of tuples (label, count) for each pattern
    :param patterns: the patterns mined by LCM
    :param label_dict: the label dictionary
    :return: the list of tuples (label, count) for each pattern
    """
    patterns_label_count = []
    for index, row in patterns.iterrows():
        label_count = {label: 0 for label in label_dict.values()}
        items = row['itemset']
        items = list(map(lambda x: (x.split('_')[0], int(x.split('_')[-1])), items))
        for (label, count) in items:
            label_count[label] = max(label_count[label], count)
        support = row['support']
        patterns_label_count.append((label_count, support))
    return patterns_label_count


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
                graphs.append((graph, support))
            elif line.startswith('v'):
                _, node_id, label = line.strip().split(' ')
                graph.add_node(int(node_id), label=label_dict[int(label)])
            elif line.startswith('e'):
                _, node_id1, node_id2, label = line.strip().split(' ')
                graph.add_edge(int(node_id1), int(node_id2))
    return graphs


def gspan_count(graph, label_dict):
    """
    Count the number of nodes for each label in the graph
    :param graph: the graph
    :param label_dict: the label dictionary
    :return: the number of nodes for each label in the graph as a dictionary
    """
    label_count = {label: 0 for label in label_dict.values()}
    for node in graph.nodes():
        label_count[graph.nodes[node]['label']] += 1
    return label_count


def compare_structure_and_feature(rule=1, metric='entropy', node_selection='split_top', with_neighbors=True,
                                  min_supp_ratio=0.7, dataset='mutagenicity', verbose=False):
    r"""
    Compare the structure and the feature of the patterns mined by gSpan and LCM as described in the paper
    :param rule: the rule to use
    :param metric: the metric to use
    :param node_selection: the node selection method to use
    :param with_neighbors: whether to use the neighbors of the selected nodes or not
    :param min_supp_ratio: the minimum support ratio
    :param dataset: the dataset to use
    :param verbose: whether to print the results or not
    :return: list of ratio of patterns that are similar in structure and feature
    """""
    nb_graph = number_of_graph_per_rule[rule]
    gspan_path = f'./results/{dataset}_{rule}_{metric}_{node_selection}{"_with_neighbors" if with_neighbors else ""}.txt'
    data_path = f'./results/{dataset}_{rule}_{metric}_{node_selection}{"_with_neighbors" if with_neighbors else ""}.data'
    transactions = build_subgraph_and_transactions(dataset=dataset, rule=rule, metric=metric,
                                                   node_selection=node_selection,
                                                   nb_graph=nb_graph,
                                                   with_neighbors=with_neighbors,
                                                   data_path=data_path)
    gs = gspan_mine_rule(nb_graph=nb_graph,
                         supp_ratio=min_supp_ratio, save_path=gspan_path, data_path=data_path)
    if gs is None:
        return None
    label_dict = gs.label_dict
    graphs = parse_gspan_result(
        gspan_path,
        label_dict)
    patterns = pattern_mining(transactions=transactions,
                              supp_ratio=min_supp_ratio,
                              nb_graphs=number_of_graph_per_rule[rule])
    patterns_label_count = lcm_pattern_to_count(patterns, label_dict)
    graphs_label_count = [(gspan_count(graph, label_dict), support) for (graph, support) in graphs]
    ratio_supports = []
    # Compare all combinations of patterns and graphs
    for i in range(len(patterns_label_count)):
        for j in range(len(graphs_label_count)):
            pattern_label_count, pattern_support = patterns_label_count[i]
            graph_label_count, graph_support = graphs_label_count[j]
            if all([pattern_label_count[label] == graph_label_count[label] for label in pattern_label_count.keys()]):
                # The pattern and the graph have the same label count
                # Compare the support
                support_ratio = (graph_support / nb_graph) / (pattern_support / nb_graph) * (
                    log(sum(pattern_label_count.values())))
                if verbose:
                    print(
                        f"Pattern {i} and graph {j} are similar, the structure of the pattern {pattern_label_count} "
                        f"seems to be impacted by the structure in the rule with a support ratio of {support_ratio}")
                ratio_supports.append((i, j, support_ratio, graphs[j][0]))
            else:
                support_score = (pattern_support / nb_graph) * log(sum(
                    pattern_label_count.values()))
                if support_score > 0.9 and not ratio_supports.__contains__((i, -1, support_score)):
                    if verbose:
                        print(
                            f"Pattern {i} and graph {j} are not similar, the structure of the pattern {pattern_label_count} ")
                    ratio_supports.append((i, -1, support_score, pattern_label_count))
    ratio_supports.sort(key=lambda x: x[2], reverse=True)
    # Show the graph and the pattern with the highest support
    if len(ratio_supports) == 0:
        return []
    best_pattern = patterns.iloc[ratio_supports[0][0]]['itemset']

    print(f'Best pattern: {best_pattern}')
    best_entity = ratio_supports[0]
    if ratio_supports[0][1] != -1:
        best_graph = graphs[ratio_supports[0][1]][0]
        node_labels = {node: best_graph.nodes[node]['label'] for node in best_graph.nodes()}
        nx.draw(best_graph, labels=node_labels, with_labels=True)
        plt.title(f'Best graph for rule {rule}')
        plt.show()
        print(f'Best graph: {best_graph}')
    return ratio_supports


def run_on_all_rules():
    """
    Run the comparison on all rules, used only testing
    :return: None
    """
    for i in range(0, 45):
        try:
            ratio_support = compare_structure_and_feature(i, 'entropy', 'split_top', True, 0.9, 'mutagenicity',
                                                          verbose=False)
            if len(ratio_support) > 0:
                ratio_support.sort(key=lambda x: x[2], reverse=True)
                print(ratio_support)
            else:
                print(f'No similar pattern found for rule {i}')
        except Exception as e:
            print(f'Error on rule {i}')
            raise e


#run_on_all_rules()
