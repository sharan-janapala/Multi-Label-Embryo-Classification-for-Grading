# 🚀 QUICK START GUIDE

## 3-Step Setup

### Step 1: Install

```bash
conda create -n embryo_ai python=3.10 -y
conda activate embryo_ai
conda install pytorch torchvision cpuonly -c pytorch -y
pip install -r requirements.txt
```

### Step 2: Add Model

Place `EfficientNet-B0_best.pth` in the same folder as `app.py`

### Step 3: Run

```bash
streamlit run app.py
```

---

## 🎯 Two Modes

### 🤖 AI Mode
- Upload image only
- AI predicts scores + grade
- Use for new embryos

### ✋ Manual Mode  
- Input scores + upload image
- Uses your scores for grade
- Use for validation

---

## 💡 Tips

1. **First time**: Use Manual Mode with your CSV data to validate
2. **Production**: Use AI Mode for automatic prediction
3. **Speed**: First prediction slower (model loads)
4. **Downloads**: Get CSV report + annotated image

---

## ⚠️ Need Model File?

The model file `EfficientNet-B0_best.pth` is trained using your pipeline (Blocks 1-2).

Copy it from:
```
/content/drive/MyDrive/Embryo_classification/Outputs/trained_models/EfficientNet-B0_best.pth
```

---

**You're ready! Run `streamlit run app.py`** 🚀
