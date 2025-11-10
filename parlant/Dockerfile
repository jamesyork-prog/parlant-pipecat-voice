FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/app_tools
COPY main.py /app/main.py

# Create data directory for agent_id.txt
RUN mkdir -p /app/data

# Set environment variable for agent_id.txt path
ENV AGENT_ID_PATH=/app/data/agent_id.txt

EXPOSE 8800

CMD ["python", "main.py"]
