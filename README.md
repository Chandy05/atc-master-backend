# ATC Master Backend

This is the FastAPI backend for the ATC Master project, deployed on Render.com.

## Features

- Visual RAG (Retrieval-Augmented Generation) using Pinecone vector database
- Google Gemini AI for exam question answering
- Upload PDFs to Pinecone (see upload_to_pinecone.py)

## Environment Variables

Set these in Render.com dashboard or local .env file:

- `PINECONE_API_KEY`: Your Pinecone API key
- `GOOGLE_API_KEY`: Your Google Gemini API key
