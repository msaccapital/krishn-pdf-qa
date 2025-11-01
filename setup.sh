#!/bin/bash
pip install transformers torch accelerate bitsandbytes sentence-transformers pypdf PyPDF2 flask flask-cors gunicorn python-dotenv numpy requests
pip install faiss-cpu --no-binary :all:
