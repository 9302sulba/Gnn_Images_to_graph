# GNN-Based Regression model on Microstructure Graphs

This project performs **regression** using **Graph Neural Networks (GNNs)** to predict material properties (e.g., temperature, Cr, Co) from graph representations of microstructure images.

---

## 🔹 1. Graph Generation

Microstructure images are converted into graph format. Each pixel or region becomes a node, and edges are created based on similarity or spatial proximity (including diagonals).

- **Input**: Grayscale microstructure images  
- **Output**: Graphs with node features and edge attributes (`.pt` or `.graphml`)
- **Node Features**: Intensity values representing Fe concentrations  
- **Edge Weights**: Absolute differences between neighboring pixel values

Tools used:
- `igraph`
- `PyTorch Geometric`
- `numpy`, `pandas`, `opencv`

---

## 🔹 2. GraphSAGE for Regression

We use the **GraphSAGE** architecture to learn from graphs and perform **multi-target regression**.

### Targets:
- `temperature`
- `Cr` (Chromium concentration)
- `Co` (Cobalt concentration)

### Model Workflow:
1. Load graph data using `PyG`
2. Normalize `temperature` using MinMaxScaler
3. Train GraphSAGE model with regression loss (MAE)
4. Evaluate and plot results (e.g., parity plots, R²)

---



