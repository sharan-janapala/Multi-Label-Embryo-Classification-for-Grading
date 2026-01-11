# Trained Model Outputs & Results

---

## 📂 Trained Model Outputs & Results (External Storage)

Due to GitHub file size limitations, large artifacts generated during model training and evaluation are hosted externally on Google Drive.

🔗 **Access the complete model training outputs and results here:**  
👉 [Google Drive – Model Training Results](https://drive.google.com/drive/folders/1aej9qM55r1h_l1YDKeybd0S0MNg9ZmNJ?usp=sharing)

---

### Contents of the Google Drive Folder
The linked folder includes:
- Trained model checkpoints
- Validation and test predictions
- Performance visualizations (accuracy, loss curves)
- Sample inference results
- Experiment-related artifacts and logs

---

### Model Summary
- **Architecture:** EfficientNet-B0
- **Task:** Multi-label embryo classification
- **Output Labels:** Expansion (EXP), Inner Cell Mass (ICM), Trophectoderm (TE)
- **Framework:** PyTorch

---

### Purpose of External Hosting
Model outputs and experimental artifacts are stored externally to:
- Maintain a lightweight and clean GitHub repository
- Enable easy access to full experimental results
- Support reproducibility and transparent evaluation

---

This document summarizes the training process, evaluation metrics, and final outputs of the **Multi-Label Embryo Classification for Grading** model.

---

## 🔬 Model Architecture

- **Backbone:** EfficientNet-B0
- **Framework:** PyTorch
- **Task Type:** Multi-label classification
- **Output Labels:**  
  - Expansion (EXP)  
  - Inner Cell Mass (ICM)  
  - Trophectoderm (TE)

Each embryo image is assigned independent grades for EXP, ICM, and TE.

---

## 📊 Training Configuration

- **Input Resolution:** 224 × 224
- **Loss Function:** Binary Cross-Entropy with Logits
- **Optimizer:** Adam
- **Learning Rate:** Tuned during experimentation
- **Batch Size:** Optimized for GPU memory
- **Epochs:** Trained until validation convergence

---

## 🧪 Model Evaluation

The model was evaluated using:
- Validation accuracy per label
- Precision and recall across grading categories
- Confidence score distribution analysis

The trained model demonstrates **stable convergence** and **consistent predictions** across embryo grading classes.

---

## 📦 Model Artifacts

Due to GitHub file size limits, trained model weights are stored externally.

- **Model File:** `EfficientNet-B0_best.pth`
- **Storage:** GitHub Large File Storage (LFS)

📁 Model training logs, checkpoints, and performance plots are available here:  
🔗 **Google Drive – Model Training & Results**  
👉 https://drive.google.com/drive/folders/1csZTUybdpujH5j-JiCOjJVjFyhsWtwwu?usp=sharing

---

## ✅ Key Outcomes

- Successfully trained a multi-label embryo grading model
- Achieved reliable performance across all grading dimensions
- Model integrated seamlessly into a Streamlit inference pipeline

---

## ⚠️ Disclaimer

This model is intended for **research and educational purposes only** and is **not approved for clinical use**.
