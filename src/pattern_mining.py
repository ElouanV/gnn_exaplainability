import numpy as np

from build_transaction import build_transaction
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from rich import print

def pattern_frequency(path, metric, rule, dataset_name, graph_ids=[], fixed_size=False, size=None, sparsity=0.5):
    print("Building transactions...")
    df = build_transaction(path, metric, rule, dataset_name, graph_ids, fixed_size, size, sparsity)
    print('Transactions built.')
    # Convert the transaction counts to binary values (0 and 1)
    df_bin = df.applymap(lambda x: 1 if x >= 1 else 0)

    # Use the Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(df_bin, min_support=0.2, use_colnames=True)

    # Generate association rules from the frequent itemsets
    a_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    # Print the frequent itemsets and association rules
    # #print("Frequent Itemsets:")
    # print(frequent_itemsets)
    # #print("\nAssociation Rules:")
    # print(a_rules)
    return frequent_itemsets, a_rules


# pattern_frequency('results/mutagenicity/gcn/gstarx', metric="entropy", rule=23,
#                   dataset_name="mutagenicity", graph_ids=np.arange(0, 373), fixed_size=True, size=3, sparsity=0.5)