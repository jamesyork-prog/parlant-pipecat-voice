#!/bin/bash

# Database Backup Script
# This script creates a timestamped backup of the PostgreSQL database
# Requirements: 8.2 - Backup existing data before migration

set -e  # Exit on error

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/whizdb_backup_${TIMESTAMP}.sql"

# Database credentials - detect current credentials from running container
# This script works with BOTH old and new credentials
if docker ps --format '{{.Names}}' | grep -q "^whiz-database$" 2>/dev/null; then
    # Get credentials from running container
    DB_NAME=$(docker exec whiz-database env | grep "^POSTGRES_DB=" | cut -d'=' -f2)
    DB_USER=$(docker exec whiz-database env | grep "^POSTGRES_USER=" | cut -d'=' -f2)
    DB_HOST="${POSTGRES_HOST:-localhost}"
    DB_PORT="${POSTGRES_PORT:-5432}"
else
    # Fallback to .env or defaults
    DB_HOST="${POSTGRES_HOST:-localhost}"
    DB_PORT="${POSTGRES_PORT:-5432}"
    DB_NAME="${POSTGRES_DB:-WhizDB}"
    DB_USER="${POSTGRES_USER:-admin}"
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Database Backup Script ===${NC}"
echo "Timestamp: ${TIMESTAMP}"
echo "Database: ${DB_NAME}"
echo "Host: ${DB_HOST}:${DB_PORT}"
echo "User: ${DB_USER}"
echo ""

# Create backup directory if it doesn't exist
if [ ! -d "${BACKUP_DIR}" ]; then
    echo -e "${YELLOW}Creating backup directory: ${BACKUP_DIR}${NC}"
    mkdir -p "${BACKUP_DIR}"
fi

# Check if running in Docker environment
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo -e "${YELLOW}Running inside Docker container${NC}"
    DOCKER_MODE=true
else
    echo -e "${YELLOW}Running on host machine${NC}"
    DOCKER_MODE=false
fi

# Perform backup
echo -e "${YELLOW}Starting backup...${NC}"

if [ "$DOCKER_MODE" = true ]; then
    # Running inside container - use pg_dump directly
    PGPASSWORD="${POSTGRES_PASSWORD:-whiz}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        -f "${BACKUP_FILE}"
else
    # Running on host - use docker exec
    CONTAINER_NAME="whiz-database"
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Error: PostgreSQL container '${CONTAINER_NAME}' is not running${NC}"
        echo "Start it with: docker-compose up -d postgres"
        exit 1
    fi
    
    echo "Using Docker container: ${CONTAINER_NAME}"
    docker exec "${CONTAINER_NAME}" pg_dump \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges > "${BACKUP_FILE}" 2>&1
    
    # Check if docker exec succeeded
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: pg_dump failed inside container${NC}"
        cat "${BACKUP_FILE}"
        rm -f "${BACKUP_FILE}"
        exit 1
    fi
fi

# Check if backup was successful
if [ $? -eq 0 ] && [ -f "${BACKUP_FILE}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo -e "${GREEN}✓ Backup completed successfully!${NC}"
    echo ""
    echo "Backup location: ${BACKUP_FILE}"
    echo "Backup size: ${BACKUP_SIZE}"
    echo ""
    echo -e "${GREEN}To restore this backup, run:${NC}"
    if [ "$DOCKER_MODE" = true ]; then
        echo "  PGPASSWORD=\${POSTGRES_PASSWORD} psql -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME} -f ${BACKUP_FILE}"
    else
        echo "  docker exec -i ${CONTAINER_NAME} psql -U ${DB_USER} -d ${DB_NAME} < ${BACKUP_FILE}"
    fi
    echo ""
    
    # List recent backups
    echo -e "${YELLOW}Recent backups:${NC}"
    ls -lh "${BACKUP_DIR}" | tail -n 5
    
    exit 0
else
    echo -e "${RED}✗ Backup failed!${NC}"
    exit 1
fi
