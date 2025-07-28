# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy model files and scripts
COPY MyModel.joblib .
COPY MyEncoder.joblib .
COPY process_sections.py .
COPY requirements.txt .
COPY all-MiniLM-L6-v2/ ./all-MiniLM-L6-v2/
COPY Collection_1/ ./Collection_1/

# Install torch/transformers CPU-only versions
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers==4.40.1 sentence-transformers==2.7.0 \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "process_sections.py"]
