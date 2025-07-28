import os
import sys
import fitz
import joblib
import json
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

# CONFIG
MODEL_PATH = "MyModel.joblib"
ENCODER_PATH = "MyEncoder.joblib"
SBERT_MODEL_PATH = "all-MiniLM-L6-v2"
FONT_SIZE_THRESHOLD = 14.0
MAX_PARAS = 5
MAX_SECTIONS = 7

# Load Models
clf = joblib.load(MODEL_PATH)
enc = joblib.load(ENCODER_PATH)
cols = clf.feature_names_in_
model = SentenceTransformer(SBERT_MODEL_PATH)
print("✅ Model and encoder loaded. Beginning PDF processing...")

# Feature Extraction
def extract_features(text, fs, bold, centered, italic, page, x0):
    return [
        fs, int(bold), int(centered), int(italic), page, x0,
        len(text), len(text.strip()), len(text.split()),
        int(any(c.isdigit() for c in text)),
        int(":" in text),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        int(any(k in text.lower() for k in ['introduction', 'chapter'])),
        int(text.isupper()),
        3 if fs >= 22 else 2 if fs >= 18 else 1 if fs >= 14 else 0
    ]

# PDF Processing
def process_pdf(path):
    print(f"🔍 Reading PDF: {path}")
    try:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return [], []

        doc = fitz.open(path)
        headings = []
        all_lines = []
        current_heading = None

        for pg in range(len(doc)):
            page = doc.load_page(pg)
            blocks = page.get_text("dict")["blocks"]

            for blk in blocks:
                if "lines" not in blk:
                    continue
                for ln in blk["lines"]:
                    line_text = " ".join(sp["text"].strip() for sp in ln["spans"]).strip()
                    if len(line_text) < 4:
                        continue
                    fs = max(sp["size"] for sp in ln["spans"])
                    font = ln["spans"][0]["font"].lower()
                    x0 = ln["spans"][0]["bbox"][0]
                    is_bold = "bold" in font
                    is_italic = "italic" in font
                    is_centered = abs(x0 - (page.rect.width / 2)) < 50

                    feat = extract_features(line_text, fs, is_bold, is_centered, is_italic, pg + 1, x0)
                    df = pd.DataFrame([feat], columns=cols)
                    label = enc.inverse_transform(clf.predict(df))[0]

                    all_lines.append({
                        "text": line_text,
                        "page": pg + 1
                    })

                    if label != "None" and fs >= FONT_SIZE_THRESHOLD:
                        if current_heading and current_heading["page"] == pg + 1 and abs(current_heading["font_size"] - fs) <= 0.5:
                            current_heading["text"] += " " + line_text
                        else:
                            if current_heading:
                                headings.append({
                                    "document": os.path.basename(path),
                                    "section_title": current_heading["text"].strip(),
                                    "page_number": current_heading["page"]
                                })
                            current_heading = {
                                "text": line_text,
                                "font_size": fs,
                                "page": pg + 1
                            }

        if current_heading:
            headings.append({
                "document": os.path.basename(path),
                "section_title": current_heading["text"].strip(),
                "page_number": current_heading["page"]
            })

        return headings, all_lines

    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")
        return [], []

# Batch PDF Threaded
def process_all_pdfs(pdf_list, pdf_dir):
    sections, paras = [], []
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(process_pdf, os.path.join(pdf_dir, f)): f for f in pdf_list}
        for fut in futures:
            s, p = fut.result()
            sections.extend(s)
            paras.extend(p)
    return sections, paras

# Ranking and Subsection Extraction
def rank_and_extract(task, sections, all_paras):
    if not sections:
        print("⚠️ No valid sections extracted. Skipping ranking.")
        return [], []

    task_emb = model.encode(task, convert_to_tensor=True)
    titles = [s["section_title"] for s in sections]
    title_embs = model.encode(titles, convert_to_tensor=True)
    sims = util.cos_sim(task_emb, title_embs).squeeze()
    top_sections = sorted(zip(sims, sections), key=lambda x: x[0], reverse=True)[:MAX_SECTIONS]

    results, subsection_analysis = [], []
    used = set()

    for rank, (score, sec) in enumerate(top_sections, start=1):
        results.append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": rank,
            "page_number": sec["page_number"]
        })

        cands = [
            p["text"] for p in all_paras
            if (p["page"] == sec["page_number"] or p["page"] == sec["page_number"] + 1)
            and sec["section_title"] not in p["text"]
            and p["text"] not in used
        ]

        if not cands:
            continue

        cand_embs = model.encode(cands, convert_to_tensor=True)
        para_sims = util.cos_sim(task_emb, cand_embs).squeeze()
        best_ids = para_sims.argsort(descending=True)[:MAX_PARAS]
        selected = [cands[i] for i in best_ids if para_sims[i] > 0.3]

        for s in selected:
            used.add(s)

        if selected:
            subsection_analysis.append({
                "document": sec["document"],
                "refined_text": "\n".join(selected),
                "page_number": sec["page_number"]
            })

    return results, subsection_analysis

# Per-Collection Processor
def process_collection(folder_name):
    print(f"\n📂 Processing: {folder_name}")
    start_time = time.time()

    json_path = os.path.join(folder_name, "challenge1b_input.json")
    pdf_dir = os.path.join(folder_name, "pdfs")
    output_path = os.path.join(folder_name, "challenge1b_output.json")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    persona = data["persona"]["role"]
    task = data["job_to_be_done"]["task"]
    pdfs = [doc["filename"] for doc in data["documents"]]

    secs, paras = process_all_pdfs(pdfs, pdf_dir)
    ranked_secs, sub_analysis = rank_and_extract(task, secs, paras)

    output = {
        "metadata": {
            "input_documents": pdfs,
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": ranked_secs,
        "subsection_analysis": sub_analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"✅ {folder_name} processed in {elapsed:.2f} seconds")

# Main: Single Collection Based on Arg
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Usage: python process_collection.py <Collection_Folder_Name>")
        sys.exit(1)

    collection_name = sys.argv[1]
    if not os.path.isdir(collection_name):
        print(f"❌ Folder '{collection_name}' not found.")
        sys.exit(1)

    process_collection(collection_name)