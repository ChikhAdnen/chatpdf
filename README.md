# Llama3_RAG
Ollama / streamlit 
### Features
 * PDF upload and processing
 * Question answering using local language models
 * PDF viewer with zoom functionality
 * Vector database for efficient information retrieval
 * Multi-query retrieval for improved accuracy
### Installation
 1- Clone the repository :
 ```
   git clone https://github.com/ChikhAdnen/chatpdf.git
   cd chatpdf
 ```
 2- Create virtual environment :
 ```
   python -m venv ollama
   source ollama/bin/activat
```
 3- Install dependencies :
 ```
    pip install -r requirements.txt
```
 4- Download ollama and pull models :
  - *https://ollama.com/download*
```
  ollama pull nomic-embed-text
  ollama pull llama3
```
 5- Run the application :
  ```
  streamlit run app.py
```
