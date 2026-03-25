# Exploring Multimodality in Federated Survival Analisys
## Master's Thesis in Computer Science and Engineering, Politecnico di Milano, academic year 2025-2026

Author: Sofia Perini

Advisor: prof. Matteo Matteucci

Co-Advisors: Alberto Archetti, Cristian Sbrolli


**Abstract:** Multimodal survival analysis integrates diverse medical data, such as whole slide images and genomic profiles, to improve prognostic accuracy in healthcare. However, data privacy regulations prevent the centralisation of patient data across institutions, limiting the applicability of conventional approaches. Federated learning addresses this challenge by enabling collaborative model training across distributed data silos without sharing raw data. This thesis presents an investigation into the integration of federated learning with multimodal survival analysis. This work builds upon a state-of-the-art early fusion architecture, adapting the training process for horizontally federated environments and implementing three federated algorithms: FedAvg, FedProx, and SCAFFOLD.

Results demonstrate that federated learning consistently outperforms isolated training across all datasets and approaches or exceeds centralised performance in balanced settings. Among federated algorithms, SCAFFOLD emerges as the most robust, achieving the narrowest confidence intervals and performances most closely aligned with the centralised ideal. This work establishes the first benchmarks for federated multimodal survival analysis, providing a foundation for future privacy-preserving medical machine learning research.

### Experiments
Currently, no studies in the existing literature have investigated the application of federated learning to multimodal survival analysis. Addressing this research gap constitutes the principal objective of this thesis, which aims to explore this integration and establish benchmarks for future work. This research builds upon stateof- the-art methodologies in multimodal survival analysis, adapting the training process to accommodate a horizontally federated environment. Three principal federated learning algorithms were incorporated: FedAvg, FedProx, and SCAFFOLD. Four datasets from TCGA were utilised, each comprising clinical data, WSIs, and genetic information for individual patients. Patient cohorts were partitioned across separate clients to simulate federated settings. To evaluate the model performance, a pipeline was implemented. Initially, hyperparameter tuning was conducted using identical data permutations and client partitions. Subsequently, the optimal hyperparameters identified through this process were applied across multiple data permutations to assess model robustness, with median and mean performance metrics computed across these different configurations. Model performance was evaluated using the concordance index (c-index) metric. The benchmarks obtained from the federated learning experiments were subsequently compared against non-federated training configurations, specifically Centralised and isolated (Islands) settings.

### Repository
This repository includes the experiment code, the notebooks to create the data splits, the notebooks for the visualisation of the results, and the data splits themselves. The raw datasets are not included due to the large size of some CSV files.

## Model
This work and the model were built on top of SurvPath, a state-of-the-art multimodal survival analysis model; the original code can be found here https://github.com/mahmoodlab/SurvPath

_Guillaume Jaume, Anurag Vaidya, Richard J Chen, Drew FK Williamson, Paul Pu Liang, and Faisal Mahmood. Modeling dense multimodal interactions between biological pathways and histology for survival prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11579–11590, 2024._
