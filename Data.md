# Raw & Processed Data Pipeline

---

## 📂 Raw & Processed Dataset (External Storage)

Due to GitHub storage limitations, large datasets used in this project are hosted externally on Google Drive.

🔗 **Access the raw and processed dataset here:**  
👉 [Google Drive – Raw & Processed Data](https://drive.google.com/drive/folders/1cDGi9IE6QReYN3o9JuIDqNcO7jKKAMqp?usp=sharing)

---

### Dataset Contents
The Google Drive folder contains:
- Raw embryo images collected at different developmental stages
- Preprocessed and normalized image datasets
- Aligned label files for multi-label embryo grading
- Supporting metadata and sample visualizations

---

### Data Processing Overview
The dataset underwent the following steps before model training:
- Image quality filtering and cleanup
- Resizing and normalization for model compatibility
- Alignment of each image with corresponding grading labels:
  - Expansion (EXP)
  - Inner Cell Mass (ICM)
  - Trophectoderm (TE)
- Validation of label consistency and class distribution

---

### Purpose of External Hosting
The dataset is stored externally to:
- Keep the GitHub repository lightweight and manageable
- Enable access to full-scale data for reproducibility
- Allow reviewers to explore raw and processed samples independently

---

### Ethical & Usage Note
All data has been anonymized and is provided strictly for **research and educational purposes**.  
This dataset is not intended for direct clinical decision-making.

---

This document describes the dataset sources, preprocessing workflow, and data alignment steps used in the **Multi-Label Embryo Classification for Grading** project.

## ⚠️ Disclaimer

All data is anonymized and used strictly for **academic and research purposes**.
