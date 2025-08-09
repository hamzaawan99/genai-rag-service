# GenAI RAG Service

A modular Retrieval-Augmented Generation (RAG) service built with FastAPI and Streamlit. This service allows you to upload documents, generate embeddings, and chat with your documents using large language models.

## Features

- **Document Processing**: Upload and process various document formats (PDF, DOCX, TXT, etc.)
- **Multiple Embedding Models**: Support for Sentence Transformers and OpenAI embeddings
- **Vector Database**: Built-in support for FAISS and Weaviate
- **Chat Interface**: Interactive chat interface with document context
- **Modular Design**: Easily extensible architecture
- **Docker Support**: Easy deployment with Docker Compose

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- OpenAI API key (optional, for using OpenAI models)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/genai-rag-service.git
   cd genai-rag-service
   ```

2. Copy the example environment file and update with your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:8501
   - Weaviate UI: http://localhost:8080
   - API Documentation: http://localhost:8000/api/docs

## Configuration

Edit the `.env` file to configure the application. Key settings include:

- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI models)
- `VECTOR_DB`: Choose between "faiss" (default) or "weaviate"
- `DEFAULT_LLM_MODEL`: Default language model to use (e.g., "gpt-3.5-turbo")
- `UPLOAD_DIR`: Directory to store uploaded files

## Project Structure

```
genai-rag-service/
├── backend/                 # Backend FastAPI application
│   ├── app/                
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Pydantic models
│   │   └── services/       # Business logic
│   └── main.py             # FastAPI application entry point
├── frontend/               # Streamlit frontend
│   └── app.py              # Streamlit application
├── config/                 # Configuration files
├── data/                   # Data directory (created at runtime)
│   ├── uploads/            # Uploaded documents
│   └── faiss_index/        # FAISS index files
├── .env.example            # Example environment variables
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile.backend      # Backend Dockerfile
├── Dockerfile.frontend     # Frontend Dockerfile
└── requirements.txt        # Python dependencies
```

## API Endpoints

### Documents

- `POST /api/v1/documents/upload` - Upload a document
- `GET /api/v1/documents/{document_id}` - Get a document by ID
- `GET /api/v1/documents/` - List all documents
- `DELETE /api/v1/documents/{document_id}` - Delete a document

### Chat

- `POST /api/v1/chat/completions` - Generate a chat completion
- `GET /api/v1/chat/models` - List available models

## Development

### Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Start the backend:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

4. Start the frontend in a new terminal:
   ```bash
   cd frontend
   streamlit run app.py
   ```

### Running Tests

```bash
pytest
```

## Deployment

### Production

For production deployment, it's recommended to:

1. Set `ENVIRONMENT=production` in your `.env` file
2. Configure a reverse proxy (e.g., Nginx) in front of the backend
3. Use a process manager like PM2 or Supervisor
4. Set up proper monitoring and logging

### Kubernetes

Kubernetes deployment files will be added in a future release.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Weaviate](https://weaviate.io/)
- [OpenAI](https://openai.com/)