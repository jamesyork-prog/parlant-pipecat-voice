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

# Copy patch script and apply it to fix Gemini model references (optional - may fail if parlant structure changes)
COPY patch_gemini_model.py /tmp/patch_gemini_model.py
RUN python /tmp/patch_gemini_model.py || echo "⚠️  Patch script failed - continuing anyway" && rm -f /tmp/patch_gemini_model.py

# Copy application code
COPY . /app/app_tools

# Copy main.py and startup script to /app root
COPY main.py /app/main.py
COPY start_servers.sh /app/start_servers.sh

# Create data directory for agent_id.txt
RUN mkdir -p /app/data

# Make startup script executable
RUN chmod +x /app/start_servers.sh

# Set environment variable for agent_id.txt path
ENV AGENT_ID_PATH=/app/data/agent_id.txt

# Expose both Parlant (8800) and Webhook (8801) ports
EXPOSE 8800 8801

CMD ["/app/app_tools/start_servers.sh"]
