import os

import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

import parse_active
from HNruleExplainer import HNRuleExplainer
from dataset.datasets import get_dataset, get_dataloader
from model.gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils
from torch_geometric.data import Data
from model.model_selector import model_selector
from ego_graph_dataset import EgoGraphDataset
from torch_geometric.data import DataLoader
import numpy as np

IS_FRESH = False


@hydra.main(config_path="config", config_name="config")
def main(config):
    # Set config
    cwd = os.path.dirname(os.path.abspath(__file__))
    config.datasets.dataset_root = os.path.join(cwd, "datasets")
    config.models.gnn_saving_path = os.path.join(cwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = config.explainers.explainer_name

    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    # Load dataset
    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    dataloader_params = {
        "batch_size": config.models.param.batch_size,
        "random_split_flag": config.datasets.random_split_flag,
        "data_split_ratio": config.datasets.data_split_ratio,
        "seed": config.datasets.seed,
    }
    dataloader = get_dataloader(dataset, **dataloader_params)
    test_indices = dataloader["test"].dataset.indices
    if config.datasets.data_explain_cutoff > 0:
        test_indices = test_indices[: config.datasets.data_explain_cutoff]

    # Load model
    model, checkpoint = model_selector("GNN", config.datasets.dataset_name, pretrained=True, return_checkpoint=True)

    # state_dict = torch.load(
    #     os.path.join(
    #         config.models.gnn_saving_path,
    #         config.datasets.dataset_name,
    #         f"{config.models.gnn_name}_"
    #         f"{len(config.models.param.gnn_latent_dim)}l_best.pth",
    #     )
    # )["net"]

    # model.load_state_dict(state_dict)

    model.to(device)

    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )

    check_dir(explanation_saving_path)
    # Test prediction accuracy and get average payoff (probability)
    ''' preds = []
    rst = []
    for data in dataloader["test"]:
        data.to(device)
        pred = model(data).softmax(-1)
        preds += [pred]
        rst += [pred.argmax(-1) == data.y]
    preds = torch.concat(preds)
    rst = torch.concat(rst)
    payoff_avg = preds.mean(0).tolist()
    acc = rst.float().mean().item()
    logger.debug("Predicted prob: " + ",".join([f"{p:.4f}" for p in payoff_avg]))
        logger.debug(f"Test acc: {acc:.4f}")'''
    rules = [23] if config.datasets.dataset_name == "mutagenicity" else [54]
    error_id = []
    for rule in rules:
        selected_graphs = [17, 18, 45, 72, 80, 81, 85, 89, 90, 92, 99, 140, 146, 149, 150, 158, 159, 181, 185, 196, 197, 198, 288, 313, 326, 328, 345, 406, 425, 434, 438, 458, 459, 460, 461, 489, 546, 551, 561, 562, 563, 564, 569, 588, 589, 605, 619, 656, 658, 685, 702, 718, 734, 802, 805, 814, 833, 851, 862, 872, 908, 928, 968, 969, 985, 988, 1044, 1049, 1068, 1082, 1084, 1097, 1130, 1131, 1178, 1194, 1206, 1218, 1219, 1225, 1242, 1243, 1244, 1261, 1278, 1293, 1317, 1318, 1374, 1379, 1382, 1383, 1408, 1410, 1412, 1423, 1436, 1437, 1468, 1470, 1480, 1483, 1486, 1491, 1576, 1588, 1596, 1606, 1661, 1674, 1690, 1704, 1705, 1711, 1712, 1713, 1714, 1772, 1775, 1776, 1777, 1782, 1785, 1790, 1791, 1802, 1803, 1804, 1813, 1858, 1859, 1860, 1861, 1874, 1912, 1915, 1916, 1937, 2011, 2014, 2027, 2029, 2031, 2043, 2049, 2052, 2070, 2079, 2101, 2194, 2217, 2219, 2235, 2285, 2372, 2387, 2405, 2410, 2418, 2420, 2446, 2489, 2492, 2493, 2499, 2501, 2507, 2526, 2541, 2542, 2543, 2547, 2565]
        ego_graph_dataset = EgoGraphDataset(
            f"/home/elouan/epita/lre/lre/first_GNN/lrde/optimal_transport/optimal_transport_for_gnn/activ_ego/mutag_{rule}labels_egos.txt",
            selected_graphs)
        # Build a dataload for the ego graph dataset
        dataloader = DataLoader(ego_graph_dataset, batch_size=1, shuffle=False)
        test_indices = dataloader.dataset.indices

        for i in tqdm(selected_graphs):
            df_result = pd.DataFrame(
                columns=["sum", "entropy", "cosine", "cheb", "likelyhood", "likelyhood_max", "hamming",
                         "focal_loss"])

            data = ego_graph_dataset[i]
            graph_id = i
            for metric in ["sum", "entropy", "cosine", "cheb", "likelyhood", "likelyhood_max", "hamming",
                           "focal_loss"]:  # add 'lin'
                if config.datasets.dataset_name == "mutagenicity":
                    dataset_name = 'mutag'
                else:
                    dataset_name = config.datasets.dataset_name
                targeted_rule = parse_active.get_rule_info(
                    f"./lrde/optimal_transport/optimal_transport_for_gnn/activ_ego/{dataset_name}_{rule}labels_egos.txt")
                targeted_class = 0
                explainer = HNRuleExplainer(model=model,
                                            dataset=dataset,
                                            dataset_name=config.datasets.dataset_name,
                                            targeted_rule=targeted_rule,
                                            targeted_class=targeted_class,
                                            device=device,
                                            max_sample_size=config.explainers.param.max_sample_size,
                                            tau=config.explainers.param.tau,
                                            subgraph_building_method=config.explainers.param.subgraph_building_method,
                                            metric=metric)

                plot_utils = PlotUtils(config.datasets.dataset_name, is_show=True)
                scores_list = []

                data = data.to(device)
                explained_example_path = os.path.join(
                    explanation_saving_path, f"example_{config.datasets.dataset_name}_{rule}_{metric}_{graph_id}.pt"
                )

                # use data.num_nodes as num_samples
                node_scores = explainer.explain(
                    data,
                    superadditive_ext=config.explainers.superadditive_ext,
                    sample_method=config.explainers.sample_method,
                    num_samples=config.explainers.num_samples,
                    k=config.explainers.num_hops,
                )
                if node_scores is None:
                    error_id += [(graph_id, len(data.x))]
                    continue

                torch.save(node_scores, explained_example_path)
                df_result[metric] = pd.Series(node_scores)
                scores_list += [node_scores]
                # print(node_scores)
                if False:
                    logger.debug(f"Plotting example {rule}.")
                    from utils import (
                        scores2coalition,
                        evaluate_coalition,
                        fidelity_normalize_and_harmonic_mean,
                        to_networkx,
                    )

                    coalition = scores2coalition(node_scores, config.explainers.sparsity)
                    title_sentence = f"Explanation {config.datasets.dataset_name} rule {rule} with {metric} for graph {graph_id}:"

                    node_attrs = data.x.tolist() if data.num_nodes == 1 else data.x
                    data['label'] = node_attrs
                    explained_example_plot_path = os.path.join(explanation_saving_path,
                                                               f"example_{config.datasets.dataset_name}_{rule}_{metric}_{graph_id}_without_edges.png")
                    plot_utils.plot(
                        to_networkx(data, to_undirected=True, node_attrs=['label', 'center']),
                        coalition,
                        x=data.x,
                        words='',
                        title_sentence=title_sentence,
                        figname=explained_example_plot_path,
                    )
                    # SAve the result dataframe as csv

                data.to('cpu')
            df_result.to_csv(
                os.path.join(explanation_saving_path,
                             f"result_{config.datasets.dataset_name}_{rule}_{graph_id}.csv"),
                index=False)
    print(error_id)


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=gstarx")
    main()
