# GNN Rule Explanability with Cooperative Games

Welcome to the GNN Rule Explanability with Cooperative Games project! This repository is inspired by the GStarX project and focuses on enhancing the explainability of Graph Neural Networks (GNNs) in the context of cooperative games. In this research internship, you will have the opportunity to explore GNN interpretability techniques and their application in understanding the decision-making process of GNN models.
This repository is still in development and will be updated regularly. If you have any questions or suggestions, feel free to contact us.
## Project Overview

The GNN Rule Explanability with Cooperative Games project aims to investigate and develop novel explainability techniques for GNNs in the context of cooperative game scenarios. Cooperative games involve multiple agents working together towards a common goal, and understanding the rules and dynamics governing their interactions is crucial. By leveraging graph neural networks and cooperative game theory, we aim to provide interpretable explanations for the decision rules learned by GNN models in these settings.
This repository contains code from [GSpan](https://github.com/betterenvi/gSpan) but we modified it to fit to our needs.
For the implementation of Hamiache-Navarro value, we highly inspired from [GStarX](https://github.com/shichangzh/gstarx) that implement this solution for GNN explainability at instance-level.



## Repository Structure

The repository is structured as follows: (It does not  follow this architecture yet, but it will be updated soon)
```angular2htmldata/ # Directory for storing datasets
├── checkpoints/ # Directory for storing trained models
├── notebooks/ # Directory for Jupyter notebooks
├── src/ # Directory for Python source code
│ ├── gspan_mine/ # Source code of gSpan
│ ├── model/ # Implementation of GNN models
│ ├── dataset/ # Code for datasets
├── README.md # Project overview and instructions (you are here)
└── requirements.txt # Required Python packages 
```


## Getting Started

To get started with the project, follow these instructions:

1. Clone the repository to your local machine using the following command:
   ```bash
   git clone git@github.com:ElouanV/gnn_exaplainability.git
   ```
   
2. Navigate to the project directory:
    ```bash
      cd gnn_exaplainability
   ```

3. Set up a virtual environment for the project:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Explore the notebooks/ directory, which contains Jupyter notebooks demonstrating different aspects of GNN explainability. These notebooks provide a starting point for your research and development.



# Contribution Guidelines
If you find any issues or have ideas for improvements, we encourage you to contribute to the project. Please follow these guidelines for contributing:

Fork the repository and create a new branch for your contribution.

Make your changes and test them thoroughly.

Create a pull request with a detailed description of your changes, including any relevant information for reviewers.

After review, your changes may be merged into the main repository.

We appreciate your contributions and look forward to your innovative ideas!