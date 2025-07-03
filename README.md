## Notebook Summaries

---

### 🔬 `main.ipynb` – Benchmarking Feature Sets for PCE Prediction

This notebook systematically benchmarks a variety of **molecular descriptors** and **learning algorithms** for predicting power-conversion efficiency (PCE) of donor–acceptor pairs from the HOPV15 dataset.

#### 🛠️ 1. Setup
- Loads dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `rdkit`, `shap`
- Cleans low-quality entries and applies stratified data splitting to ensure balanced distributions across folds

#### 🧠 2. Core Utility Functions
- `main_loop_with_stratified_cv`: Performs stratified k-fold CV with diagnostic outputs
- `combine_and_rank_feature_importance`: Fuses and ranks XGBoost & RF feature importances
- `scramble_y`: Shuffles PCE labels to verify model validity (y-randomisation)

#### ⚗️ 3. Experiments by Descriptor Type
Each section benchmarks a specific set of descriptors using Random Forest or XGBoost:
- LUMO only
- LUMO + HOMO
- MACCS + LUMO
- LUMO + MACCS (top 50 / 30 / 10 / 5 features)
- HOMO + LUMO + MACCS, Estate, Morgan, PubChem subsets + Acceptor Type
- y-randomisation sanity check

#### 🧪 4. Training and Evaluation Workflow
- Default model: **Random Forest**, switchable to **XGBoost**
- Reusable loop with scaling, cross-validation, test holdout
- Metrics: MAE, RMSE, R² (mean ± std over CV), final independent test score

#### 📊 5. Interpretability and Error Diagnostics
- SHAP summary & dependence plots for model explainability
- Error percentile plots for difficult samples
- y-randomisation ensures learning is from signal, not noise

---

### 🧬 `Plot.ipynb` – Visualizing Donor Substructures

This notebook explores donor molecule structures to identify **common substructures** associated with higher-performing OPV materials.

#### 🔍 Key Features
- Loads donor SMILES from dataset
- Extracts frequent substructures using RDKit SMARTS and fingerprints
- Visualizes fragments as:
  - Molecule structure images
  - Substructure frequency bar charts
  - Grids of substructure-highlighted donors

#### 🎯 Applications
- Useful for connecting structure to PCE
- Enables interpretation of feature importance in conjunction with SHAP results

#### 🔗 Integration
- Pair with `main.ipynb` for:
  - Correlating top SHAP features with actual chemical fragments
  - Validating ML-driven feature importance with visual evidence

---

### ✅ TL;DR
- `main.ipynb`: Benchmarks ~10 descriptor combinations for OPV PCE prediction; interprets models with SHAP and sanity checks via y-randomisation
- `Plot.ipynb`: Visualizes chemical fragments in donor molecules; aids interpretation of predictive features

## License
- **Code**: MIT (see `LICENSE`)
- **Datasets & pretrained models**: CC-BY-4.0 (see `LICENSE-DATA`)
Please cite our accompanying publication when reusing.
