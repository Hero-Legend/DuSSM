# DuSSM: Disentangling Cognitive Duality in Biomedical Linguistics

[![Paper](https://img.shields.io/badge/Paper-Nature%20Communications-blue)](https://github.com/Hero-Legend/DuSSM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)

This repository contains the official implementation of **DuSSM**, a parallel State-Space framework designed to resolve the "feature entanglement" in biomedical relation extraction. By structurally isolating localized morphological triggers (Explicit Stream) from long-range semantic logic (Implicit Stream), DuSSM achieves superior clinical reliability and $O(N)$ inference efficiency.


## 🌟 Key Contributions
* **Parallel Cognitive Disentanglement**: Structurally separates morphological feature extraction from topological reasoning to resolve feature entanglement.
* **Selective State-Space Formulation**: Models biomedical text as a continuous dynamic system using the Mamba architecture, effectively capturing long-range causal dependencies.
* **Clinical Determinism**: Establishes a "determinism boundary" with **94.32% precision** on non-interaction cases, significantly mitigating "alert fatigue" in pharmacovigilance.
* **Computational Efficiency**: Leverages linear-time complexity $O(N)$ for a 1.6x inference speedup over Transformer baselines.

## 📊 Benchmark Results

DuSSM has been rigorously validated across four heterogeneous biomedical datasets:

| Dataset | Type | Metric | Result | 
| :--- | :--- | :--- | :--- | 
| **DDI-2013** | Drug-Drug Interaction | Macro F1 | **82.27%** |
| **ChemProt** | Chemical-Protein | Micro F1 | **87.64%** | 
| **GAD** | Gene-Disease | F1/AUC | **83.55%** | 
| **EU-ADR** | Gene-Disease | F1/AUC | **86.04%** | 

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/Hero-Legend/DuSSM.git](https://github.com/Hero-Legend/DuSSM.git)
cd DuSSM

# Create environment
conda create -n dussm python=3.10
conda activate dussm

# Install core dependencies
pip install torch==2.4.0 mamba-ssm causal-conv1d transformers
```

## 🚀 Usage

# 1. Data Preparation

The experimental scripts support DDI-2013, ChemProt, GAD, and EU-ADR. Place your preprocessed datasets in the data/ directory.


# 2. Training & Evaluation
To run the primary DDI extraction task:
```bash
python CNN+transformer-mamba.py --dataset ddi2013 --batch_size 32 --lr 1e-4 --epochs 22
```

To run gene-disease association experiments:
```bash
python GAD_expriment.py --fold 10
python EUADR_expriment.py --fold 10
```

##  🔍 Mechanistic Insight

DuSSM utilizes Cognitive Gating to filter linguistic noise. The selection intensity ($\Delta_t$) adaptively contracts to skip clinical jargon ($\downarrow$ 62%) and expands to record critical relational anchors.

##  📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
