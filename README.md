# 🐾 PawSense

**AI-Powered Dog Emotion Detection & Personalized Product Recommendation**

> Upload a photo of your dog — PawSense identifies the breed, detects behavioral state, and recommends the perfect products from [Chewy.com](https://www.chewy.com).

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Course** | ISOM5240 Group Project |
| **Company** | [Chewy, Inc.](https://www.chewy.com) |
| **App URL** | [pawsense.streamlit.app](https://pawsense.streamlit.app) |
| **Model URL** | [Bonnnz/CustomModel_dogemotion](https://huggingface.co/Bonnnz/CustomModel_dogemotion) |

### Business Objective (≤ 50 words)

> This project aims to detect dog behavioral states from images using deep learning, enabling personalized product recommendations and improving customer retention for pet e-commerce platforms like Chewy.com.

---

## 🏗️ Architecture

```
                ┌─ Pipeline A (image-classification) ─→ Breed: "Golden Retriever"
                │   Model: wesleyacheng/dog-breeds-multiclass-image-classification-with-vit
                │   (pretrained, no fine-tuning)
                │
User uploads ───┤
  dog photo     │
                └─ Pipeline B (image-classification) ─→ Behavior: "happy"
                    Model: Bonnnz/CustomModel_dogemotion
                    (fine-tuned ViT on Dog Emotion Dataset)
                                │
                                ▼
                    Breed + Behavior + Chewy Product Catalog
                                │
                                ▼
                    Pipeline C (text2text-generation) ─→ Personalized recommendation
                    Model: google/flan-t5-base
                    (prompt engineering, no fine-tuning)
```

---

## 🔬 Pipelines

### Pipeline A — Dog Breed Recognition

| Config | Detail |
|---|---|
| Task | `image-classification` |
| Model | `wesleyacheng/dog-breeds-multiclass-image-classification-with-vit` |
| Fine-tuned | No (pretrained) |
| Output | Dog breed name (120 breeds) |

### Pipeline B — Emotion Detection

| Config | Detail |
|---|---|
| Task | `image-classification` |
| Model | `Bonnnz/CustomModel_dogemotion` |
| Base model | `google/vit-base-patch16-224` |
| Fine-tuned | **Yes** (on Dog Emotion Dataset v2) |
| Labels | 4 classes: sad, angry, relaxed, happy |
| Dataset | [Dewa/Dog_Emotion_Dataset_v2](https://huggingface.co/datasets/Dewa/Dog_Emotion_Dataset_v2) (4,000 images) |

### Pipeline C — Product Recommendation

| Config | Detail |
|---|---|
| Task | `text2text-generation` |
| Model | `google/flan-t5-base` |
| Fine-tuned | No (prompt engineering) |
| Product source | Curated Chewy.com product catalog (20+ real products) |
| Output | Personalized recommendation text |

---

## 📊 Dataset

| Field | Detail |
|---|---|
| Source | [Dewa/Dog_Emotion_Dataset_v2](https://huggingface.co/datasets/Dewa/Dog_Emotion_Dataset_v2) |
| Labels | 4 classes: sad (0), angry (1), relaxed (2), happy (3) |
| Training samples | 2,500 |
| Test samples | 500 |
| Balance | Balanced across all 4 classes |

---

## 🚀 Quick Start

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/PawSense.git
cd PawSense
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy

---

## 📁 Repository Structure

```
PawSense/
├── app.py                 # Streamlit application (3 pipelines integrated)
├── requirements.txt       # Python dependencies
└── README.md              # Description
```

---

## 📂 Full Submission Structure

```
Group42_documentation/
├── Project_report.pdf
└── Experimental_results.xlsx
Group42_program/
├── Python_notebooks/
│   ├── Pipeline_A_Dog_Breed_Recognition.ipynb
│   ├── Pipeline_B_Dog_Emotion_Finetune.ipynb
│   ├── Pipeline_C_Product_Recommendation.ipynb
│   └── FullPipeline_ExperimentalResults.ipynb
├── GitHub_App_Files/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
Group42_Dataset_files/
├── data.csv
└── Fine-tuned_Model_files/
    └── CustomModel_dogemotion.zip
Group42_presentation/
├── Presentation_slide.pptx
└── grp42.mp4
Streamlit Cloud App URL
```

---

## ⚠️ Disclaimer

PawSense detects observable behavioral patterns only and does not constitute veterinary advice. Please consult a professional veterinarian for health concerns.

---

## 📜 License

This project is for academic purposes (ISOM5240). Product data referenced from [Chewy.com](https://www.chewy.com).
