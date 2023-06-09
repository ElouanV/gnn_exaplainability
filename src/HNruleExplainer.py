import torch
from torch_geometric.utils import subgraph, to_dense_adj
from utils import *
import numpy as np
from rule_evaluator import RuleEvaluator
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import to_networkx


class HNRuleExplainer(object):
    r"""
    Class to extract representative features and structures captured by a rule of a model.
    """
    def __init__(self, model, device, dataset, dataset_name, targeted_rule, targeted_class, metric='cosine',
                 max_sample_size=10, tau=0.01, subgraph_building_method="remove", edge_probs=None):
        r"""
        Args:
            model (torch.nn.Module): The model to explain.
            device (torch.device): The device on which the model is.
            dataset (torch_geometric.data.Dataset): The dataset on which the model is trained.
            dataset_name (str): The name of the dataset.
            targeted_rule (int): The index of the rule to explain.
            targeted_class (int): The index of the class to explain.
            metric (str): The metric to use to compute the similarity between the rule and the subgraph.
            max_sample_size (int): The maximum size of the subgraph to consider.
            tau (float): The threshold to consider a node as part of the subgraph.
            subgraph_building_method (str): The method to use to build the subgraph.
            edge_probs (dict): The edge probabilities to use to build the subgraph.
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_sample_size = max_sample_size
        self.coalitions = get_ordered_coalitions(max_sample_size)
        self.tau = tau
        self.M = get_associated_game_matrix_M(self.coalitions, max_sample_size, tau)
        self.M = self.M.to(device)
        self.subgraph_building_func = get_graph_build_func(subgraph_building_method)
        self.targeted_rule = targeted_rule
        self.dataset = dataset
        if edge_probs is not None:
            self.edge_probs = edge_probs
        else:
            edge = np.ones((dataset.data.x.shape[-1], dataset.data.x.shape[-1]))
            degree = np.ones((dataset.data.x.shape[-1], 20))
            self.edge_probs = {"edge_prob": edge, "degre_prob": degree}
        self.targeted_class = targeted_class
        self.metric = metric
        self.rule_evaluator = RuleEvaluator(self.model, dataset_name,
                                            (dataset.data.edge_index, dataset.data.x, dataset.data.y), targeted_rule,
                                            metric=metric,
                                            unlabeled=False, edge_probs=self.edge_probs)

    def compute_rule_scores(self, data, emb=None):
        ''' UNUSED
        Compute the score of the rule on the given graph data
        :param data: torch_geometric.data.Data graph data to compute the score
        :param emb: the embedding of the graph data
        :return: the score of the rule
        '''
        metric_value = real_value = -1
        if not self.targeted_rule:
            print("TODO")
        else:
            if emb is not None:
                score = self.rule_evaluator.compute_score_emb(emb)
            else:
                score = self.rule_evaluator.compute_score(data)
        return score, (metric_value, real_value)

    def explain(self, data, superadditive_ext=True, sample_method='khop', num_samples=10, k=3):
        '''
        Compute the HN score of the given graph data
        :param data: torch_geometric.data.Data graph data to compute the HN score
        :param superadditive_ext: whether to use the superadditive extension
        :param sample_method: the method to use to sample the subgraphs
        :param num_samples: the number of samples to use to compute the HN score
        :param k: the number of hops to use to sample the subgraphs
        :return: the scores of the nodes
        '''
        data = data.to(self.device)
        adj = (to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].detach().cpu())
        char_function = RuleEvaluator.compute_score
        if data.num_nodes <= self.max_sample_size:
            scores = self.compute_scores(data, adj, char_function, superadditive_ext)
        else:
            scores = torch.zeros(data.num_nodes)
            counts = torch.zeros(data.num_nodes)
            if sample_method == "khop" or num_samples == -1:
                num_samples = data.num_nodes

            i = 0
            while not counts.all() or i < num_samples:
                sampled_nodes, sampled_data, sampled_adj = sample_subgraph(
                    data, self.max_sample_size, sample_method, i, k, adj
                )
                sampled_scores = self.compute_scores(
                    sampled_data, sampled_adj, char_function, superadditive_ext
                )
                scores[sampled_nodes] += sampled_scores
                counts[sampled_nodes] += 1
                i += 1

            nonzero_mask = counts != 0
            scores[nonzero_mask] = scores[nonzero_mask] / counts[nonzero_mask]
        return scores.tolist()

    def compute_scores(self, data, adj, char_func, superadditive_ext):
        '''
        Compute the HN score of the given graph to targeted rule of the objet
        :param data: torch_geometric.data.Data graph data to compute the HN score
        :param adj: the adjacency matrix of the graph
        :param char_func: the characteristic function to use to compute the HN score
        :param superadditive_ext: whether to use the superadditive extension
        :return: the scores of the nodes
        '''
        n = data.num_nodes
        if n == self.max_sample_size:
            coalitions = self.coalitions
            M = self.M
        else:
            coalitions = get_ordered_coalitions(n)
            M = get_associated_game_matrix_M(coalitions, n, self.tau)
            M = M.to(self.device)

        v = self.get_coalition_payoffs(data, coalitions, self.subgraph_building_func)
        if superadditive_ext:
            v = v.tolist()
            v_ext = superadditive_extension(v, n)
            v = torch.tensor(v_ext).to(self.device)
        v = v.to(self.device)
        P = get_associated_game_matrix_P(coalitions, n, adj)
        P = P.to(self.device)
        H = torch.sparse.mm(P, torch.sparse.mm(M, P))

        H_tilde = get_limit_game_matrix(H, is_sparse=True)
        # Convert v to a float tensor
        v = v.float()
        v_tilde = torch.sparse.mm(H_tilde, v.view(-1, 1)).view(-1)

        scores = v_tilde[:n].cpu()
        return scores

    def get_coalition_payoffs(self, data, coalitions, subgraph_building_func):
        '''
        Compute the payoffs of the coalitions
        :param data: torch_geometric.data.Data graph data to compute the HN score
        :param coalitions: the coalitions to compute the payoffs
        :param subgraph_building_func: the function to use to build the subgraphs
        :return: the payoffs of the coalitions
        '''
        n = data.num_nodes
        masks = []
        for coalition in coalitions:
            mask = torch.zeros(n)
            mask[list(coalition)] = 1.0
            masks += [mask]

        coalition_mask = torch.stack(masks, axis=0)
        masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
        masked_dataloader = DataLoader(
            masked_dataset, batch_size=256, shuffle=False, num_workers=0
        )

        masked_payoff_list = []
        for masked_data in masked_dataloader:
            masked_payoff_list.append(self.compute_score_batch(masked_data))

        masked_payoffs = torch.cat(masked_payoff_list, dim=0)
        return masked_payoffs

    def compute_score_batch(self, data_batch):
        '''
        Compute the score of the given batch of subgraphs
        :param data_batch: the batch of subgraphs
        :return: the score of the batch of subgraphs
        '''
        scores = []
        for i in range(data_batch.num_graphs):
            subgraph = data_batch[i]
            # Convert the subgraph to a networkx graph
            node_attrs = subgraph.x #.tolist() if subgraph.num_nodes == 1 else subgraph.x
            subgraph['label'] = node_attrs
            graph = to_networkx(subgraph, to_undirected=True, node_attrs=['label'])

            # Compute the score for the networkx graph
            score = self.rule_evaluator.compute_score(graph)

            # Append the score to the list of scores
            scores.append(score)

        # Convert the list of scores to a PyTorch tensor
        scores_tensor = torch.tensor(scores)

        return scores_tensor
