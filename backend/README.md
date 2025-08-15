# GenAI RAG Service - Backend

## Weaviate - Vector Database

Setup the environment variable for API User and API Key for Weaviate in .env file. Check .env.sample for reference

## PostgreSQL - Database

Setup the environment variables for Port, DB, User and Password for Postgres in .env file. Check .env.sample for reference

## Redis - Cache

Setup the environment variable for Port and Password for Redis in .env file. Check .env.sample for reference

Run Weaviate, PostgreSQL and Redis in a container using Docker Compose. Docker Compose file is provided just run the command

```bash
docker compose up -d
```