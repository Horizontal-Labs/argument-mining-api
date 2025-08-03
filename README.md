# Argument Mining API

A FastAPI-based web service for argument mining and analysis.

## Features

- RESTful API for argument mining
- Health check endpoints

## Prerequisites

- Python 3.9 or higher (required for numpy 2.3.1 and pandas 2.3.0)
- pip (Python package manager)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd argument-mining-api
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root directory:
   ```bash
   # Copy the example environment file
   cp .env.example .env  # On Unix-based systems
   # OR manually create .env file
   ```

2. **Configure environment variables** in the `.env` file:
   ```env
   MODEL_API_URL=https://models.example.com/render
   TIMEOUT_SECONDS=30
   ```

## How to Run the API

### Development Mode

1. **Navigate to the project directory**:
   ```bash
   cd argument-mining-api
   ```

2. **Ensure your virtual environment is activated**

3. **Start the development server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The `--reload` flag enables auto-reload on code changes for development.

### Production Mode

For production deployment:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, you can access:

- **Interactive API Documentation (Swagger UI)**: http://localhost:8000/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8000/redoc

## Project Structure

```
argument-mining-api/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   └── api/
│       ├── routes/          # API route definitions
│       │   ├── chat.py      # Chat endpoints
│       │   └── health.py    # Health check endpoints
│       ├── schemas/         # Pydantic models for request/response
│       ├── services/        # Business logic and external integrations
│       └── utils/           # Utility functions
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port number in the uvicorn command
2. **Module not found errors**: Ensure your virtual environment is activated and dependencies are installed
3. **Environment variables not loaded**: Check that your `.env` file exists and is properly formatted
