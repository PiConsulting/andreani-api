FROM python:3.7.9

RUN apt-get update
RUN apt-get install -y tesseract-ocr ffmpeg libsm6 libxext6
WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

WORKDIR /workspace/src/
EXPOSE 8000
CMD ["uvicorn", "main:app",  "--port", "8000", "--host", "0.0.0.0", "--reload"]