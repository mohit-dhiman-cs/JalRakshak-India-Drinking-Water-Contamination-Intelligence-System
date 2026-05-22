<div align="center">
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit Logo" width="80"/>
  <h1>💧 JalRakshak Intelligence System</h1>
  <p><strong>India Drinking Water Contamination Intelligence & Machine Learning Suite</strong></p>
  <p><strong>Live:- https://jalrakshak-water-contamination.streamlit.app/</strong></p>
</div>

---

## 🌍 Overview

**JalRakshak** (meaning "Water Protector") is an advanced, interactive Machine Learning educational dashboard and diagnostic platform designed to safeguard India's drinking water infrastructure. By transforming raw chemical telemetry, citizen complaints, and geographic data into actionable intelligence, JalRakshak enables urban planners and engineers to proactively combat water contamination.

Built entirely in **Python** with a stunning **Streamlit** space-dark glassmorphic UI, this platform serves as a complete masterclass in Data Science algorithms, featuring interactive sandboxes for Regression, Classification, NLP, and Clustering models.

---

## 🚀 Key Modules & Capabilities

The intelligence suite is divided into six progressive analytical units:

### 📈 Unit I: WQI Trend Monitor (Simple Linear Regression)
* Tracks and forecasts the Water Quality Index (WQI) over temporal horizons.
* Features interactive slope ($\beta_1$) and intercept ($\beta_0$) simulators to teach the fundamental mathematics of OLS geometry.
* Displays live error metric evaluations (MSE, RMSE, MAE).

### 🧪 Unit II: Parameter Predictor (Multiple Linear Regression)
* Multi-dimensional modeling utilizing chemical sensor data (`pH`, `TDS`, `Turbidity`, `DO`, `BOD`, `Nitrates`).
* Includes a dynamic **Feature Selection Studio** to highlight multicollinearity (VIF) and parameter significance ($p$-values).
* Provides a real-time "What-If" equation simulator with live WQI gauge plotting.

### 🛡️ Unit III: Safety Classification Engine (Logistic Regression)
* Classifies water sources into binary `Safe` or `Unsafe` categories.
* Features a dynamic **Decision Threshold Studio** ($p\_{cutoff}$) to teach the balance between True Positives (Alarms) and False Positives (Panic).
* Includes stunning Plotly interactive ROC curves and Confusion Matrix heatmaps.

### 🐦 Unit IV: Outbreak Detector (NLP Naive Bayes)
* Mines and analyzes citizen SMS/Twitter complaints to detect localized gastrointestinal outbreaks.
* Features an **NLP Preprocessing Sandbox** demonstrating text normalization, stopword removal, and Porter Stemming step-by-step.
* Visualizes the heaviest token impacts and calculates outbreak probabilities via real-time speed-gauges.

### 🧩 Unit V: District Risk Clustering (K-Means)
* Unsupervised learning module that segments districts into distinct risk archetypes.
* Calculates the **Hopkins Statistic** to validate clustering tendency.
* Renders a highly interactive Plotly **3D Cluster Explorer** for exploring multi-dimensional demographic risk bounds.

### 🤖 Unit VI: AI Health Alert System (DBSCAN + GIS)
* Anomaly detection mapping using DBSCAN to identify highly anomalous municipal regions.
* Features a **GIS Mapbox** geospatial dashboard for nationwide hazard tracking.
* Integrates a **Hierarchical Proximity Dendrogram** and a live AI simulated terminal that streams automated mitigation action protocols.

---

## 💻 Tech Stack

* **Frontend / Framework:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `KMeans`, `DBSCAN`, `MultinomialNB`, `PCA`)
* **Statistical Modeling:** Statsmodels (`statsmodels.api`)
* **Visualizations:** Plotly (`plotly.express`, `plotly.graph_objects`, `plotly.figure_factory`), Matplotlib, Seaborn

---

## ⚙️ Installation & Usage

### Prerequisites
Ensure you have Python 3.10+ installed on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/jalrakshak.git
cd jalrakshak
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies include `streamlit`, `pandas`, `scikit-learn`, `statsmodels`, `plotly`, `scipy`, `nltk`)*

### 3. Launch the Application
```bash
python -m streamlit run app.py
```
*The intelligence suite will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <i>Securing India's Water Future 💧</i>
</div>
