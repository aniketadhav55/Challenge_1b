
# Challenge 1B - PDF Section Extraction using ML

This project is designed to **extract and rank relevant sections from academic or business PDFs** using machine learning models and sentence embeddings. It supports multiple collections (`Collection_1`, `Collection_2`, etc.) and is packaged with Docker for easy portability.

---

## 🧠 Project Highlights

- Uses **Decision Tree classifier** to detect section headings.
- Leverages **Sentence-BERT (MiniLM-L6-v2)** for semantic ranking.
- Extracts top **N relevant sections and paragraphs** based on a task query.
- Processes PDFs in **parallel** for fast execution.
- Accepts JSON input and writes structured JSON output.

---

## 🗂️ Folder Structure

```
.
├── Collection_1/
│   ├── pdfs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection_2/
│   └── ...
├── Collection_3/
│   └── ...
├── all-MiniLM-L6-v2/
├── MyModel.joblib
├── MyEncoder.joblib
├── process_sections.py
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Core Pipeline

### 🔹 `process_sections.py` Steps

1. **Input**: JSON containing persona, task, and PDF filenames.
2. **Parse PDFs**: Using `PyMuPDF`, extract lines with formatting features.
3. **Predict**: Use ML model and encoder to classify headings.
4. **Rank**: Use SBERT to compute similarity between task and headings.
5. **Extract Subsections**: Select top paragraphs based on semantic similarity.
6. **Output**: Structured JSON with top sections and refined content.

---

## 🐋 Docker Instructions

### 🔨 Build the Docker Image

```bash
docker build --platform=linux/amd64 -t my1b:latest .
```

### 🚀 Run the Container

```bash
docker run --rm -v ${PWD}/Collection_1:/app/Collection_1 my1b:latest python process_sections.py Collection_1
```

Replace `Collection_1` with the desired folder name (must include `challenge1b_input.json` and `pdfs/`).

---

## 📦 Requirements

Install dependencies locally (for development):

```bash
pip install -r requirements.txt
```

### Key Libraries Used

- `sentence-transformers`
- `scikit-learn`
- `PyMuPDF`
- `pandas`, `numpy`, `joblib`, `concurrent.futures`, `torch`

---

## 📤 Input JSON Format (`challenge1b_input.json`)

```json
{
  "persona": { "role": "PhD Researcher" },
  "job_to_be_done": { "task": "Study cancer biomarkers" },
  "documents": [
    { "filename": "paper1.pdf" },
    { "filename": "paper2.pdf" }
  ]
}
```

## 📥 Output JSON Format (`challenge1b_output.json`)

Includes ranked headings and refined paragraphs.

---

## 👨‍💻 Author

Nayan Dhamane  
Third Year B.E. Computer Engineering, SPPU (Batch 2026)
