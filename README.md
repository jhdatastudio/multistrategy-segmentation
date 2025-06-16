# 🧠 Multi-Strategy Customer Segmentation – Demo 5

This Streamlit app compares **three distinct customer segmentation strategies** on a transactional retail dataset. It demonstrates how different approaches — cohort analysis, churn prediction, and PCA-based persona mapping — offer complementary insights into customer behavior.

🔗 **Live App**: [Open Streamlit Demo](https://multistrategy-segmentation-yfa9zjceekmyjocx8tb7xc.streamlit.app/)
📁 **Repository**: [View on GitHub](https://github.com/jhdatastudio/multistrategy-segmentation)

---

## 🚀 Project Overview

This demo walks through and compares:

### 📆 Cohort Analysis  
- Segments customers by acquisition month  
- Tracks **retention trends** over time using heatmaps

### 🔁 Churn Prediction  
- Labels churn based on recency behavior (>90 days inactivity)  
- Trains a **logistic regression model** using RFM features  
- Evaluates performance using cross-validated **ROC AUC**

### 🧠 PCA Personas  
- Applies **PCA** on scaled RFM data  
- Visualizes customer distribution across latent behavioral axes  
- Supports persona-driven targeting and UX discovery

---

## 🧮 Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Streamlit** for interactive dashboard
- **OpenPyXL** for reading Excel files
- **Caching** to improve load times in deployment

---

## 📊 App Structure & Tabs

| Tab | Description |
|-----|-------------|
| **📆 Cohort Analysis** | Monthly retention heatmap by acquisition group |
| **🔁 Churn Prediction** | Logistic regression + cross-validation scores |
| **🧠 PCA Personas** | Visualize customer patterns using 2D PCA |
| **📚 Summary** | Comparison table + business recommendations |

---

## ✅ Business Value

- **Compare segmentation techniques** and their tradeoffs
- Understand which methods are best for different business questions:
  - Retention tracking (Cohort)
  - Risk targeting (Churn)
  - Persona discovery (PCA)

---

## 📁 Project Structure

demo-05-multistrategy-segmentation/
├── data/
│   └── Online Retail.xlsx              # Sample dataset (UCI)
├── streamlit_app_tabs.py              # Main Streamlit app (tabbed)
├── requirements.txt                   # Python dependencies
└── README.md

---

## 🌐 About JH Data Studio

JH Data Studio offers tailored data consulting services across AI, analytics, and automation — empowering businesses to make smarter decisions faster.

👉 Visit [jhdatastudio.com](https://jhdatastudio.com) to learn more or book a discovery call.
