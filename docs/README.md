<div align="center">

![Smart Loan Default Risk Assessment Platform](../header.svg)

# 📚 Documentation Index Hub

> **Central Knowledge Base for Technical Feasibility Reports, Research Specifications & Architecture Guidelines**

</div>

---

## 📌 Available Documentation Catalog

| Document Title | Type | Description | Link |
| :--- | :--- | :--- | :--- |
| **Technical Feasibility Report & PRD** | Feasibility / PRD | Evaluates system feasibility, MVP architecture, PySpark ETL pipelines, multi-table dataset joining, model training, and Streamlit/FastAPI deployment. | [`Loan_default_feasibility_report.md`](Loan_default_feasibility_report.md) |
| **Explainable Loan Default Prediction Report** | Research Paper | Comprehensive research on Big Data analytics (Apache Spark), Machine Learning (XGBoost/Random Forest), and Explainable AI (SHAP / LIME) for auditable credit risk scoring. | [`Explainable_Loan_Default_Prediction_Research_Report.md`](Explainable_Loan_Default_Prediction_Research_Report.md) |

---

## 🗂️ Documentation Directory Structure

```
BIG_DATA_PROJECT/
└── docs/
    ├── README.md                                             # Documentation Index Page (You are here)
    ├── Loan_default_feasibility_report.md                   # Technical Feasibility, PRD & Architecture Survey
    └── Explainable_Loan_Default_Prediction_Research_Report.md # Research Report on XAI & Big Data Analytics
```

---

## ✍️ How to Contribute to Documentation

We welcome documentation contributions from all team members! Follow these steps to contribute:

### 1. Branch Strategy
Always create a dedicated documentation branch off `develop`:
```bash
# Switch to develop and pull latest changes
git checkout develop
git pull origin develop

# Create a new docs branch
git checkout -b docs/your-feature-name
```

### 2. File Location & Naming Rules
- Place all new technical documentation inside the [`docs/`](file:///c:/Users/ACER/Desktop/BIG_DATA_PROJECT/docs) directory.
- Use clear, descriptive file names in `snake_case` or `Title_Case` (e.g., `data_dictionary.md`, `model_card.md`).
- Ensure images or diagrams are stored cleanly or embedded using vector SVG / markdown graphics.

### 3. Update the Index Page
Whenever you add a new report or documentation file:
1. Open [`docs/README.md`](file:///c:/Users/ACER/Desktop/BIG_DATA_PROJECT/docs/README.md).
2. Add a new row to the **Available Documentation Catalog** table with a link and summary.
3. Update the **Directory Structure** tree.

### 4. Submit Your Changes
```bash
# Stage your documentation changes
git add docs/

# Commit with conventional commit style
git commit -m "docs: add model card and dataset dictionary"

# Push branch to remote
git push -u origin docs/your-feature-name
```
Submit a **Pull Request** targeting the `develop` branch for peer review.

---

## 👥 Contributors & Acknowledgments

This documentation hub is collaboratively maintained by the **Smart Loan Default Risk Assessment Platform** team.

- **Big Data & Pipelines**: PySpark ETL, Data Ingestion & Relational Table Merging
- **Machine Learning & XAI**: Model Training (XGBoost/LightGBM) & SHAP Explanations
- **Documentation & Research**: Technical Specifications, PRD, and Research Reports

---

*Maintained with excellence.* ✨
