import pandas as pd
from tqdm import tqdm
from ego_graph_dataset import EgoGraphDataset
from torch_geometric.data import DataLoader
import torch
import parse_active
from HNruleExplainer import HNRuleExplainer
from utils import check_dir, get_logger, PlotUtils
import os
import os

import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

import parse_active
from HNruleExplainer import HNRuleExplainer
from dataset.datasets import get_dataset, get_dataloader
from utils import check_dir, get_logger, PlotUtils
from model.model_selector import model_selector
from ego_graph_dataset import EgoGraphDataset
from torch_geometric.data import DataLoader
import numpy as np

IS_FRESH = False

import warnings
from multiprocessing import pool

warnings.filterwarnings("ignore")


def process_rule(model, rule, dataset_name, dataset, metrics, device, config, explanation_saving_path, error_id):
    print(f'Running rule {rule}')
    selected_graphs = []
    ego_graph_dataset = EgoGraphDataset(
        f"./activ_ego/mutag_{rule}labels_egos.txt",
        selected_graphs)
    selected_graphs = ego_graph_dataset.index_list
    # Build a dataload for the ego graph dataset
    dataloader = DataLoader(ego_graph_dataset, batch_size=1, shuffle=False)
    if len(selected_graphs) > 6000:
        print(f'Skip rule {rule}, the number of graphs ({len(selected_graphs)} exceeds the limit (8000)')
        return
    rule_threads_pool = pool.ThreadPool(processes=30)
    for i in tqdm(selected_graphs):
        result = rule_threads_pool.apply_async(process_graph,
                                               args=(ego_graph_dataset, i, False, dataset_name, dataset, metrics,
                                                     device, model, rule, config, explanation_saving_path,
                                                     error_id))
    # Wait for all threads to finish
    result.get()
    rule_threads_pool.close()
    print(f'Finished rule {rule}')


def process_graph(ego_graph_dataset, index, plot_result=False, dataset_name=None, dataset=None,
                  metrics=['entropy'], device=None, model=None, rule=None, config=None, explanation_saving_path=None,
                  error_id=None):
    df_result = pd.DataFrame(
        columns=["sum", "entropy", "cosine", "cheb", "likelyhood", "likelyhood_max", "hamming",
                 "focal_loss"])
    data = ego_graph_dataset[index]
    graph_id = index
    for metric in metrics:
        if dataset_name == "Mutagenicity":
            dataset_name = 'mutag'
        else:
            dataset_name = dataset_name
        targeted_rule = parse_active.get_rule_info(
            f"./activ_ego/{dataset_name}_{rule}labels_egos.txt")
        targeted_class = 0
        explainer = HNRuleExplainer(model=model,
                                    dataset=dataset,
                                    dataset_name=config.datasets.dataset_name.lower(),
                                    targeted_rule=targeted_rule,
                                    targeted_class=targeted_class,
                                    device=device,
                                    max_sample_size=config.explainers.param.max_sample_size,
                                    tau=config.explainers.param.tau,
                                    subgraph_building_method=config.explainers.param.subgraph_building_method,
                                    metric=metric)

        plot_utils = PlotUtils(config.datasets.dataset_name.lower(), is_show=True)
        scores_list = []

        data = data.to(device)
        explained_example_path = os.path.join(
            explanation_saving_path, f"example_{config.datasets.dataset_name.lower()}_{rule}_{metric}_{graph_id}.pt"
        )

        # use data.num_nodes as num_samples
        try:
            node_scores = explainer.explain(
                data,
                superadditive_ext=config.explainers.superadditive_ext,
                sample_method=config.explainers.sample_method,
                num_samples=config.explainers.num_samples,
                k=config.explainers.num_hops,
            )
        except Exception as e:
            node_scores = None
            raise e
        if node_scores is None:
            error_id += [(graph_id, len(data.x))]
            continue
        # Create directory if it does not exist
        check_dir(os.path.join(explanation_saving_path, f"rule_{rule}"))
        torch.save(node_scores, explained_example_path)
        df_result[metric] = pd.Series(node_scores)
        scores_list += [node_scores]

        # Add nodes scores to the data object
        node_scores_tensor = torch.tensor(node_scores, device=device).unsqueeze(1)
        data.x = torch.cat([data.x, node_scores_tensor], dim=1)

        # Save the data object with the node scores
        explained_example_path = os.path.join(
            explanation_saving_path,
            f"rule_{rule}/{config.datasets.dataset_name.lower()}_{rule}_{metric}_{graph_id}.pt"
        )
        torch.save(data, explained_example_path)
        if plot_result:
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
            # Save the result dataframe as csv
        data.to('cpu')
    df_result.to_csv(
        os.path.join(explanation_saving_path,
                     f"rule_{rule}/result_{config.datasets.dataset_name.lower()}_{rule}_{graph_id}.csv"),
        index=False)


@hydra.main(config_path="config", config_name="config")
def main(config):
    try:
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
            device = torch.device("cuda:0")
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

        # Load model
        model, checkpoint = model_selector("GNN", config.datasets.dataset_name, pretrained=True, return_checkpoint=True)

        model.to(device)

        explanation_saving_path = os.path.join(
            config.explainers.explanation_result_path,
            config.datasets.dataset_name.lower(),
            config.models.gnn_name,
            explainer_name,
        )

        check_dir(explanation_saving_path)

        rules = np.arange(55, 56)
        error_id = []
        metrics = ['entropy']
        thread_pool = pool.ThreadPool(10)
        for rule in rules:
            result = thread_pool.apply_async(process_rule, args=(
            model, rule, config.datasets.dataset_name, dataset, metrics, device, config, explanation_saving_path, error_id))

        result.get()
        thread_pool.close()
        print(error_id)
    except Exception as e:
        with open('error_report.txt', 'w') as f:
            f.write(str(e))
            raise e
        exit()


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=hnrule")
    main()
