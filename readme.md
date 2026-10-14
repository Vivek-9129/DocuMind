
# DocuMind — Chat with PDF

An AI-powered PDF question-answering application built using
Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF documents
- Extract and chunk document text
- Generate semantic embeddings
- Retrieve relevant passages using FAISS
- Generate context-grounded answers using Groq
- View source passages with page references

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq API

## Run Locally

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```



## Limitations

Scanned PDFs without extractable text may require OCR.
Answer quality depends on document text extraction and retrieval.
