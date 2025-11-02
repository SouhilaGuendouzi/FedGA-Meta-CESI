# 🧠 FedGA-Meta: Genetic and Meta-Learning Aggregation for Federated Learning in Industrial Cyber-Physical Systems

## 📘 Overview
**FedGA-Meta** is a hierarchical federated learning (FL) framework designed for **Industrial Cyber-Physical Systems (ICPS)**.  
It tackles key FL challenges such as **data heterogeneity**, **domain shift**, **limited communication resources**, and **participant variability**.  

The framework integrates:
- **Genetic Algorithms (FedGA)** for adaptive and efficient model aggregation,  
- **Model-Agnostic Meta-Learning (MAML)** to enhance generalization to new participants,  
- **CORAL (CORrelation ALignment)** for gradient alignment and domain adaptation,  
- And a **hierarchical architecture** (Edge → Fog → Cloud) that optimizes computation and communication.

## ⚙️ Architecture
FedGA-Meta relies on a three-layer hierarchy:

### 🧩 Edge Layer
- Local collaborators (robots, sensors, IoT devices, etc.) perform local training on private data.  
- Each edge model is periodically sent to its assigned **fog server**.

### ☁️ Fog Layer
- Each fog server aggregates local edge models using an **enhanced Genetic Algorithm (FedGA)**.  
- Applies the **inner-update** phase of MAML on its benchmark subset.  
- Sends **gradients of base layers** (not full models) to the cloud to reduce communication cost.

### 🌐 Cloud Layer
- Aligns gradients across fog domains using **CORAL alignment**.  
- Applies the **outer-update** phase of MAML for global adaptation.  
- Broadcasts the updated global base model back to all fog servers.

## 🧬 Workflow Summary
1. Initialize the global model at the cloud and distribute to fog servers.  
2. Each fog integrates the global base model with its local extractor.  
3. Edge collaborators train locally for several epochs.  
4. Fog servers aggregate local models via **Enhanced FedGA**.  
5. Fog servers perform **MAML inner-update** and send gradients.  
6. Cloud performs **CORAL alignment** and **MAML outer-update**.  
7. Updated global weights are redistributed to all fogs.  

## 🧠 Algorithms
FedGA-Meta includes the following routines:

- **EdgeUpdate** — local learning at each participant  
- **FogAggregation** — partial aggregation using FedGA + MAML inner-update  
- **CloudAggregation** — global aggregation using CORAL + MAML outer-update  
- **FedGA-Meta (Main Routine)** — orchestrates the full FL workflow  

Each step is formally defined in the implementation and algorithms (see Algorithms 2–5 in the manuscript).

## 🧪 Experiments
The framework was evaluated on five benchmark datasets to simulate **domain-shifted** FL environments:

| Fog | Dataset  | Feature Space | Type  |
|-----|-----------|---------------|-------|
| f1  | MNIST     | 28×28×1       | Grayscale |
| f2  | USPS      | 16×16×1       | Grayscale |
| f3  | SVHN      | 32×32×3       | RGB |
| f4  | EMNIST    | 28×28×1       | Grayscale |
| f5  | MNIST-M   | 28×28×3       | RGB |

- Each fog manages **10 edge participants** (50 total).  
- Datasets were partitioned with a **Dirichlet α = 0.5** to simulate **Non-IID** distributions.  
- Training was conducted over **50 FL rounds** using a workstation with:
  - NVIDIA RTX 3090 GPU  
  - Intel Core i9 CPU  
  - 32 GB RAM  

### 🔍 Evaluation Metrics
- Test Loss (cross-entropy)  
- Accuracy  
- F1-Score  
- Expected Calibration Error (ECE)  
- Worst 10 % Accuracy (Fairness indicator)

### 📊 Comparative Frameworks
FedGA-Meta was compared to:
- FedAvg  
- FedPer  
- FedProx  
- FedMAML  
- FedGA  

### 🚀 Results
FedGA-Meta outperforms all baselines in:
- **Adaptability** — higher accuracy & F1 under domain shift.  
- **Generalization** — smooth adaptation to late-joining participants.  
- **Cost-effectiveness** — better trade-off between local computation and communication.  


## 🧱 Technologies & Dependencies
FedGA-Meta is implemented in **Python 3.9** and leverages several scientific and machine learning libraries to ensure modularity, flexibility, and scalability of experiments.

### Core Libraries
- **torch** — Deep learning framework for model definition, training, and gradient computation.  
- **pygad** — Genetic algorithm optimization for partial aggregation at the fog layer.  
- **tsne-torch** — Visualization of high-dimensional model embeddings and feature distributions.  
- **numpy** — Numerical computation and tensor manipulation.  
- **pandas** — Dataset management, preprocessing, and statistical analysis.  
- **scikit-learn** — Data splitting, normalization, and evaluation metrics.  
- **scipy** — Scientific computations and numerical optimization utilities.  
- **seaborn** — Statistical visualization of metrics and distributions.  
- **matplotlib** — Plotting and result visualization for training curves and heatmaps.
  

### Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage


### 🔹 Run all experiments automatically
To execute all comparative frameworks (FedAvg, FedProx, FedPer, FedMAML, FedGA, and FedGA-Meta) under different local training configurations, simply run:


#### On Windows:
```bash
lunch-project.bat
```

#### On MacOS:
```bash
chmod +x run_all.sh
./run_all.sh
```
### 🔹 Visualize results

After the training phase is completed, all results (metrics, plots, and LaTeX tables) are automatically saved inside:
```bash
results_algos/
```

To visualize the metrics as figures (e.g., Accuracy, F1-score, ECE, and fairness), run:
```bash
python visualise_metrics.py

```

To display or export the summary tables in LaTeX or terminal format, run:
```bash
python visualise_tabs.py

```
