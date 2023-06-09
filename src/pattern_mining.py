from build_transaction import build_transaction
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import utils
from ego_graph_dataset import select_active_graph
from tqdm import tqdm
import pandas as pd
import os
from utils import scores2coalition
from utils import to_networkx

from gspan_mine.gspan_mining.config import parser
from gspan_mine.gspan_mining.main import main


number_of_graph_per_rule = utils.MUTAGENICITY_NUMBER_OF_GRAPH_PER_RULE
def pattern_frequency(path, metric, rule, dataset_name, graph_ids=[], fixed_size=False, size=None, sparsity=0.5,method='split_top'):
    print("Building transactions...")
    df = build_transaction(path, metric, rule, dataset_name, graph_ids, fixed_size, size, sparsity,method=method)
    print('Transactions built.')
    # Convert the transaction counts to binary values (0 and 1)
    df_bin = df.applymap(lambda x: 1 if x >= 1 else 0)

    # Use the Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(df_bin, min_support=0.2, use_colnames=True)

    # Generate association rules from the frequent itemsets
    a_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    return frequent_itemsets, a_rules


def build_graph_data(dataset='mutagenicity' ,rule=1, metric='entropy', node_selection='fixed_size', nb_graph=2923):
    graphs = select_active_graph(f'./activ_ego/mutag_{rule}labels_egos.txt', 2,0,[])
    skipped_index = []
    with open(f'results/{dataset}_{rule}_{metric}_{node_selection}.data', 'w+') as f:
        for i in tqdm(range(nb_graph)):
            graph = to_networkx(graphs[i], to_undirected=True, node_attrs=['center', 'x'])
            df_node_score = pd.read_csv(os.path.join("./results/mutagenicity/gcn/gstarx",
                             f"rule_{rule}/result_{dataset}_{rule}_{i}.csv"))
            if df_node_score is None or len(df_node_score) == 0:
                skipped_index.append(i)
                continue
            node_score = df_node_score[metric].values
            coalition = scores2coalition(node_score, sparsity=0.5, fixed_size=True, size=3)
            # If the node that has the label 'center' to True in the graph is not in the coalition, add it
            for node in graph.nodes():
                if graph.nodes[node]['center'] and node not in coalition:
                    coalition.append(node)
            # Select the subgraph induce by the coalition
            subgraph = graph.subgraph(coalition)
            f.write(f't # {i}\n')
            for node in subgraph.nodes():
                f.write(f'v {int(node)} {graph.nodes[node]["x"]}\n')
            for edge in subgraph.edges():
                f.write(f'e {int(edge[0])} {int(edge[1])} 0\n')
    print(f"Skipped {len(skipped_index)} graphs")

    # Build the graph.data file for a given rule
def build_graph_data_neighbors(dataset='mutagenicity', rule=1, metric='entropy', node_selection='fixed_size',
                                   nb_graph=2923):
    graphs = select_active_graph(f'./activ_ego/mutag_{rule}labels_egos.txt', 2, 0, [])
    skipped_index = []
    with open(f'results/{dataset}_{rule}_{metric}_{node_selection}_with_neighbors.data', 'w+') as f:
        for i in tqdm(range(nb_graph)):
            try:
                graph = to_networkx(graphs[i], to_undirected=True, node_attrs=['center', 'x'])
            except Exception as e:
                print(e)
                skipped_index.append(i)
                continue
            df_node_score = pd.read_csv(os.path.join("./results/mutagenicity/gcn/gstarx",
                                                         f"rule_{rule}/result_{dataset}_{rule}_{i}.csv"))
            if df_node_score is None or len(df_node_score) == 0:
                skipped_index.append(i)
                continue
            node_score = df_node_score[metric].values
            coalition = scores2coalition(node_score, sparsity=0.5, fixed_size=True, size=3)
                # If the node that has the label 'center' to True in the graph is not in the coalition, add it
            for node in graph.nodes():
                if graph.nodes[node]['center'] and node not in coalition:
                    coalition.append(node)
                # Select the subgraph induce by the coalition
                # Add direct neighbors of the coalition to the coalition without adding nodes that are already in the coalition
            for node in coalition:
                for neighbor in graph.neighbors(node):
                    if neighbor not in coalition:
                        coalition.append(neighbor)
            subgraph = graph.subgraph(coalition)
            f.write(f't # {i}\n')
            for node in subgraph.nodes():
                f.write(f'v {int(node)} {graph.nodes[node]["x"]}\n')
            for edge in subgraph.edges():
                f.write(f'e {int(edge[0])} {int(edge[1])} 0\n')
    print(f"Skipped {len(skipped_index)} graphs")


def gspan_mine_rule(rule=1, metric='entropy', node_selection='split_top', with_neighbors=False, supp_ratio=0.9,
                            dataset='mutagenicity'):
        if with_neighbors:
            build_graph_data_neighbors(dataset, rule, metric, node_selection,
                                           nb_graph=number_of_graph_per_rule[rule])
        else:
            build_graph_data(dataset, rule, metric, node_selection, nb_graph=number_of_graph_per_rule[rule])
        nb_graph = number_of_graph_per_rule[rule]
        min_support = int(nb_graph * supp_ratio)
        args_str = f'-s {min_support} -p True -d False ./results/{dataset}_{rule}_{metric}_{node_selection}{"_with_neighbors" if with_neighbors else ""}.data'
        FLAGS, _ = parser.parse_known_args(args=args_str.split())
        gs = main(FLAGS)
        return gs