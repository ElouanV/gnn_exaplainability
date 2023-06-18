import networkx

from collections import defaultdict
from torch import as_tensor
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse

import torch
import numpy as np

from torch.nn.functional import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RuleEvaluator:
    def __init__(self, model_to_explain, dataset_name, datas, target_rule, metric="cosine", atom_labels=True,
                 unlabeled=False, edge_probs=None):
        self.gnnNets = model_to_explain

        self.dataset = dataset_name
        self.target_rule_name = target_rule
        self.rules = self.load_rules(dataset_name)
        self.target_rule = self.rules[target_rule]
        self.metric = metric
        if datas is not None:
            self.atoms = get_atoms(dataset_name, datas, unlabeled)
            self.revatoms = {v: k for k, v in self.atoms.items()}
        self.unlabeled = unlabeled
        self.len_vects = datas[1].shape[-1]
        self.funcs = {
            "sum": self.sum_score,
            "lin": self.linear_score,
            "entropy": self.cross_entropy_score,
            "cheb": self.cheb_score,
            "cosine": self.cosine,
            "likelyhood": self.log_likelyhood,  # change to max likelyhood
            "likelyhood_max": self.max_likelyhood,  # change to max likelyhood

            "hamming": self.hamming_score,
            "focal_loss": self.focal_loss,
        }
        if edge_probs is not None:
            self.edge_probs = edge_probs["edge_prob"]
            self.degre_prob = edge_probs["degre_prob"]

        self.atom_labels = atom_labels

    def get_layer(self):
        return self.target_rule[0]

    def compute_score(self, graph, metric=None):
        """
        graph is an NX_graph
        :param graph:
        :return:
        """

        emb = self.get_embedding(graph, self.get_layer())
        return self.compute_score_emb(emb, metric)

    def compute_score_adj(self, adj, feat):
        """
        graph is an NX_graph
        :param graph:
        :return:
        """

        emb = self.get_embedding_adj(adj, feat, self.get_layer())
        return self.compute_score_emb(emb, self.metric)

    def compute_score_emb(self, emb, metric=None):
        if metric is None:
            return self.funcs[self.metric](emb, self.target_rule)
        else:
            return self.funcs[metric](emb, self.target_rule)
        return score

    def compute_feature_matrix(self, graph):
        if self.dataset == "ba2":
            return torch.ones(len(graph), 10) * 0.1
        if self.unlabeled:
            return torch.ones((len(graph), self.len_vects)) / self.len_vects
        if self.atom_labels:
            indices = []
            labels = networkx.get_node_attributes(graph, 'label')
            for node in graph.nodes():
                a = labels[node]
                l = [self.atoms[k] for k in self.atoms.keys()]
                debug = list(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))
                index = next(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))
                indices.append(index)

            index_tensor = as_tensor(indices)
            return one_hot(index_tensor, len(self.atoms.keys()))
        else:
            indices = []
            labels = networkx.get_node_attributes(graph, 'label')
            for node in graph.nodes():
                # index = next(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))

                indices.append(self.revatoms[graph.nodes[0]["label"]])

            index_tensor = torch.as_tensor(indices)
            return one_hot(index_tensor, len(self.atoms.keys()))  # len(self.atoms.keys()))

    def get_embedding(self, graph, layer):
        X = self.compute_feature_matrix(graph).type(torch.float32).to(device)
        A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
        A = dense_to_sparse(A)[0].to(device)
        embeddinng = self.gnnNets.embeddings(X, A)[layer][0]
        return embeddinng

    def get_output(self, graph):
        X = self.compute_feature_matrix(graph).type(torch.float32)
        A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
        A = dense_to_sparse(A)[0]
        output = self.gnnNets.forward(X, A)
        return output

    def get_embedding_adj(self, adj, feat, layer):

        adj = dense_to_sparse(adj)[0]
        embeddinng = self.gnnNets.embeddings(feat, adj)[layer][0]
        return embeddinng

    def get_embedding_dense(self, adj, feat, layer):

        # adj = dense_to_sparse(adj)[0]
        embeddinng = self.gnnNets.embeddings(feat, adj)[layer][0]
        return embeddinng

    def sum_score(self, embedding, target_rule):
        alpha = 0.1

        emb = torch.clamp(embedding, 0, 1)

        mask = torch.zeros_like(emb)
        mask[target_rule[1]] = 1
        ratio = (sum(mask) / len(mask))
        mask = mask.to('cpu')
        mask.apply_(lambda x: (1 - ratio) if x == 1 else -(ratio))
        mask = mask.to('cuda:0')
        return compute_score_with_coefficients(embedding, coefficients=mask)

    def cosine(self, embedding, target_rule):
        emb = torch.clamp(embedding, 0, 1)

        mask = torch.zeros_like(emb)
        mask[target_rule[1]] = 1
        return cosine_similarity(emb, mask, dim=0).item()

    def cheb_score(self, embedding, target_rule, p=2):

        mask = torch.zeros_like(embedding)
        mask[target_rule[1]] = 1
        return -sum(map(lambda x: torch.abs(x[0] - x[1]) ** p, zip(mask, embedding))).item() ** (1 / p)

    def compute_coefficient(self, target_rule):
        layer, target_components = target_rule
        number_of_components = 20
        #    max(map(lambda xo: max(x[1]), filter(lambda x: x[0] == layer, self.rules))) + 1
        coefficients = torch.zeros(number_of_components)
        for _, rule in filter(lambda x: x[0] == layer, self.rules):
            for component in rule:
                if component in target_components:
                    coefficients[component] += 1
                else:
                    coefficients[component] -= 1
        return coefficients / number_of_components

    def linear_score(self, embedding, target_rule):
        coefficients = self.compute_coefficient(target_rule)
        return compute_score_with_coefficients(embedding, coefficients=coefficients)

    def cross_entropy_score(self, embedding, target_rule):

        emb = torch.clamp(embedding, 0, 1)

        components = emb[target_rule[1]] + 1E-10
        res = sum(map(lambda x: torch.log(x), components))
        # Bad example: mas = 0!
        # maybe it would be good to use the output of the compute_score_with_coefficients as weights for cross_entropy
        return res.item()

    def focal_loss(self, embedding, target_rule, alpha=0.25, gamma=2, threshold=0.5):
        emb = torch.clamp(embedding, 0, 1)

        ce = self.cross_entropy_score(emb, target_rule)
        pt = emb
        focal_loss = alpha * (1 - pt) ** gamma * ce
        return focal_loss.sum().item()

    def max_likelyhood(self, embedding, target_rule):
        emb = torch.clamp(embedding, 0, 1)
        rules = [el for el in self.rules if el[0] == target_rule[0]]
        # target_id = rules.index(target_rule)
        probs = [self.log_likelyhood(emb, rule) for rule in rules if rule != target_rule]
        # probs = [self.log_likelyhood(emb, rule) for rule in rules]

        return self.log_likelyhood(emb, target_rule) - max(probs)

    def log_likelyhood(self, emb, rule):
        components = emb[rule[1]] + 1E-5
        return components.log().sum().item()



    def discrete_max_likelyhood(self, embedding, target_rule):
        emb = torch.clamp(embedding, 0, 1)
        emb = emb > 0

        rules = [el for el in self.rules if el[0] == target_rule[0]]
        probs = [(emb[rule[1]] > 0).sum() / len(rule[1]) for rule in rules]

        return ((emb[target_rule[1]] > 0).sum() / len(target_rule[1])) / (max(probs) + 1E-5)

    def hamming_score(self, embedding, target_rule):

        # emb = emb>0
        mask = torch.zeros_like(embedding)
        mask[target_rule[1]] = 1
        return -sum(map(lambda x: 0 if int(bool(x[0])) == x[1] else 1, zip(embedding, mask)))

    def activate(self, emb):
        return (emb[self.target_rule[1]].prod() > 0).item()

    @staticmethod
    def activate_static(target_rule, emb):
        return (emb[target_rule[1]].prod() > 0).item()
        # return np.product([emb[i]>0 for i in self.target_rule[1]])>0.5

    def real_score(self, graph):
        if (graph.number_of_edges() > 0):
            indexes = [self.revatoms[graph.nodes[el]["label"]] for el in graph.nodes]
            edge_p = np.log(
                [self.edge_probs[indexes[u], indexes[v]] for u, v in graph.edges]).sum() / graph.number_of_edges()

            degre_p = np.log([self.degre_prob[self.revatoms[graph.nodes[n]["label"]], len(graph.adj[n])] for n in
                              graph.nodes]).sum() / graph.number_of_nodes()
            return edge_p + degre_p
        else:
            return -10

    @staticmethod
    def load_rules(dataset):
        names = {"ba2": ("ba2"),
                 "aids": ("Aids"),
                 "BBBP": ("Bbbp"),
                 "Mutagenicity": ("Mutag"),
                 "mutag" : ("Mutag"),
                 "mutagenicity": ("Mutag"),
                 "DD": ("DD"),
                 "PROTEINS_full": ("Proteins")
                 }

        name = names[dataset]

        # file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
        file = "./datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
        # file = "/home/ata/ENS de Lyon/Internship/Projects/MCTS/inter-compres/INSIDE-GNN/data/Aids/Aids_activation_encode_motifs.csv"
        rules = list()
        with open(file, "r") as f:
            for l in f:
                r = l.split("=")[1].split(" \n")[0]
                label = int(l.split(" ")[3].split(":")[1])
                rule_no = int(l.split(' ')[2])
                rules.append((label, r, rule_no))
        out = dict()
        for label, r, no in rules:
            c = r.split(" ")
            layer = int(c[0][2])
            components = list()
            for el in c:
                components.append(int(el[5:]))
            out[(layer, label, no)] = (layer, components)

        return out



def get_atoms(dataset,datas, unlabeled=False):
    atoms_aids = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co",
                    11: "Br",
                    12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K",
                           21: "Pd",
                           22: "Au",
                           23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn",
                           32: "Ga", 33: "Hg",
                           34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}
    atoms_mutag = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                            11: "K", 12: "Li", 13: "Ca"}
    BBBPs= ["C", "N", "O", "S", "P", "BR", "B", "F", "CL", "I", "H", "NA", "CA"]
    atoms_bbbp= {i: v  for i, v in enumerate(BBBPs) }
    all_atoms = {"mutag": atoms_mutag, "aids": atoms_aids,"BBBP": atoms_bbbp, 'Mutagenicity': atoms_mutag, 'mutagenicity': atoms_mutag}

    if unlabeled:
        return {0:0}
    if dataset in all_atoms.keys() and False: # I add a and False so that it always goes to the else, and that's waht we need here
        return all_atoms[dataset]
    else:
        n_atoms = datas[1].shape[-1]
        return {i: i for i in range(n_atoms)}


def compute_score_with_coefficients(embedding, coefficients):
    emb = torch.clamp(embedding, 0, 1)
    sum = torch.dot(emb, coefficients).item()
    return sum
