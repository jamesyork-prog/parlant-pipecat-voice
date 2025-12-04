# Utility Scripts

This directory contains utility scripts for repository management, database operations, and maintenance tasks.

## Repository Rename Orchestration

### Main Orchestration Script

**Location:** `scripts/orchestrate_rename.py`

**Purpose:** Orchestrates the complete repository rename process from `old-repo-name` to `new-repo-name` in three automated phases: audit, update, and validation. This is the recommended way to perform the repository rename as it coordinates all the individual utilities and provides comprehensive error handling.

**Quick Start:**
```bash
# Interactive mode (recommended)
./scripts/run_orchestration.sh

# Non-interactive mode (for automation)
./scripts/run_orchestration.sh --no-interactive
```

**Features:**
- ✅ Three-phase orchestration (audit → update → validation)
- ✅ Interactive prompts for user confirmation at each phase
- ✅ Automatic backup creation with timestamps
- ✅ Comprehensive error handling and rollback mechanisms
- ✅ Git history verification
- ✅ Docker configuration validation
- ✅ Detailed progress reporting
- ✅ Automatic cleanup option

**The Three Phases:**

1. **Phase 1: Audit**
   - Scans entire codebase for old repository name
   - Generates detailed report with file paths and line numbers
   - Shows summary of files and occurrences found
   - Saves `audit_report.txt` for review

2. **Phase 2: Update**
   - Creates timestamped backup directory
   - Updates all identified files
   - Validates syntax for each file type
   - Saves `update_report.txt` with results
   - Offers rollback on failure

3. **Phase 3: Validation**
   - Re-runs audit to verify no old references remain
   - Validates Docker Compose configuration
   - Verifies Git history integrity (commit count, SHA)
   - Tests git blame functionality
   - Provides comprehensive validation summary

**Usage:**

```bash
# Interactive mode with prompts (recommended)
python3 scripts/orchestrate_rename.py

# Non-interactive mode (no prompts)
python3 scripts/orchestrate_rename.py --no-interactive

# Save results to JSON
python3 scripts/orchestrate_rename.py --output results.json

# Custom repository names
python3 scripts/orchestrate_rename.py \
  --old-name old-repo-name \
  --new-name new-repo-name

# Custom backup directory
python3 scripts/orchestrate_rename.py \
  --backup-dir /path/to/backups
```

**Shell Wrapper:**
```bash
# The wrapper checks dependencies and provides helpful messages
./scripts/run_orchestration.sh --help
```

**What Gets Updated:**
- Docker Compose service names and references
- Container names in scripts and documentation
- Repository URLs in documentation
- Configuration file references
- All hardcoded string literals

**Safety Features:**
- Automatic backups before any changes
- Syntax validation prevents breaking files
- Rollback capability on errors
- Git history preservation verification
- Interactive confirmation at each phase
- Detailed logging and reporting

**Output Files:**
- `audit_report.txt` - Detailed audit results
- `update_report.txt` - Update operation results
- `.backup_rename_YYYYMMDD_HHMMSS/` - Backup directory
- `results.json` - Complete results (if --output specified)

**Next Steps After Orchestration:**
1. Review changes with `git diff`
2. Test Docker services: `docker-compose up -d`
3. Rename repository on Git hosting service
4. Update local Git remote URL
5. Commit and push changes
6. Remove backup directory if successful

**Requirements Satisfied:**
- **Requirement 1.1:** Repository rename orchestration
- **Requirement 2.1:** Identify all occurrences
- **Requirement 2.2:** Update Docker service names
- **Requirement 2.3:** Update script container names
- **Requirement 2.4:** Update configuration files

**See Also:**
- `scripts/ORCHESTRATION_README.md` - Detailed orchestration documentation
- Individual utility scripts below for standalone usage

---

## Repository Audit Scripts

### Audit Repository References

**Location:** `scripts/audit_repository_references.py`

**Purpose:** Searches the entire codebase for occurrences of the old repository name and generates a detailed audit report. This is essential for the repository rename process to ensure all references are identified and updated.

**Quick Start:**
```bash
# Display audit report in terminal
./scripts/run_audit.sh

# Save report to file
./scripts/run_audit.sh --save

# Generate JSON report
./scripts/run_audit.sh --json --save
```

**Features:**
- ✅ Recursive file search with configurable patterns
- ✅ Binary file detection and exclusion
- ✅ Common ignore patterns (.git, __pycache__, node_modules, etc.)
- ✅ Detailed context (2 lines before/after each match)
- ✅ Multiple output formats (text and JSON)
- ✅ File path, line number, and content reporting
- ✅ Summary statistics (files scanned, matches found, etc.)

**Direct Usage (Python script):**
```bash
# Basic usage
python3 scripts/audit_repository_references.py

# Custom search term
python3 scripts/audit_repository_references.py --search "old-name"

# Save to file
python3 scripts/audit_repository_references.py --output audit.txt

# JSON format
python3 scripts/audit_repository_references.py --format json --output audit.json

# More context lines
python3 scripts/audit_repository_references.py --context 5

# Additional ignore patterns
python3 scripts/audit_repository_references.py --ignore "*.log" --ignore "temp"
```

**Wrapper Script Usage:**
```bash
# Show help
./scripts/run_audit.sh --help

# Display text report
./scripts/run_audit.sh

# Save text report to audit_report.txt
./scripts/run_audit.sh --save

# Save JSON report to audit_report.json
./scripts/run_audit.sh --json --save

# Save to custom file
./scripts/run_audit.sh --output my_audit.txt
```

**Output Format (Text):**
```
================================================================================
REPOSITORY REFERENCE AUDIT REPORT
================================================================================

Search Term: old-repo-name
Root Directory: /path/to/repo
Files Scanned: 197
Files Skipped: 8
Files with Errors: 0

Total Matches: 20
Files with Matches: 7

MATCHES:
--------------------------------------------------------------------------------

File: README.md
Matches: 2

  Line 80:
    1. **Clone the repository**
       ```bash
  >    git clone https://github.com/yourusername/old-repo-name.git
       cd old-repo-name
       ```
```

**Output Format (JSON):**
```json
{
  "search_term": "old-repo-name",
  "root_directory": "/path/to/repo",
  "files_scanned": 197,
  "files_skipped": 8,
  "files_with_errors": 0,
  "total_matches": 20,
  "files_with_matches": 7,
  "matches": [
    {
      "file_path": "README.md",
      "line_number": 80,
      "line_content": "   git clone https://github.com/yourusername/old-repo-name.git",
      "context_before": ["1. **Clone the repository**", "   ```bash"],
      "context_after": ["   cd old-repo-name", "   ```"],
      "old_reference": "old-repo-name"
    }
  ],
  "skipped_files": ["data/binary.dat", "..."],
  "error_files": []
}
```

**Excluded Patterns:**
The audit automatically excludes:
- `.git/` - Git repository metadata
- `__pycache__/` - Python bytecode cache
- `node_modules/` - Node.js dependencies
- `.pytest_cache/` - Pytest cache
- `venv/`, `env/` - Virtual environments
- `dist/`, `build/` - Build artifacts
- Binary files (`.pyc`, `.so`, `.dll`, images, archives, etc.)

**Use Cases:**
1. **Pre-rename audit:** Identify all references before starting the rename
2. **Post-rename verification:** Confirm all references were updated
3. **Documentation:** Generate reports for review and approval
4. **Automation:** Integrate into CI/CD pipelines

**Requirements Satisfied:**
- **Requirement 2.1:** Identify all occurrences of old repository name

---

### Update Repository References

**Location:** `scripts/update_repository_references.py`

**Purpose:** Updates all occurrences of the old repository name with the new name, with comprehensive backup, rollback, and validation capabilities. This ensures safe and reversible updates to the codebase.

**Quick Start:**
```bash
# Dry run to preview changes (recommended first step)
./scripts/run_update.sh --dry-run --files docker-compose.yml README.md

# Update specific files
./scripts/run_update.sh --files docker-compose.yml scripts/README.md

# Rollback if needed
./scripts/run_update.sh --rollback .backup_20240101_120000

# Cleanup backup after successful update
./scripts/run_update.sh --cleanup .backup_20240101_120000
```

**Features:**
- ✅ String replacement with formatting preservation
- ✅ Automatic backup of all modified files before changes
- ✅ Rollback functionality to restore original files
- ✅ Syntax validation for Python, JSON, YAML, Shell, and Markdown files
- ✅ Dry-run mode for safe testing
- ✅ Detailed update reports with statistics
- ✅ Error handling with failed update tracking

**Direct Usage (Python script):**
```bash
# Dry run (preview changes without modifying files)
python3 scripts/update_repository_references.py --dry-run --files file1.txt file2.py

# Update specific files
python3 scripts/update_repository_references.py --files docker-compose.yml README.md

# Custom old/new names
python3 scripts/update_repository_references.py --old-name "old-repo" --new-name "new-repo" --files file.txt

# Specify backup directory
python3 scripts/update_repository_references.py --backup-dir ./my_backup --files file.txt

# Rollback from backup
python3 scripts/update_repository_references.py --rollback .backup_20240101_120000

# Cleanup backup directory
python3 scripts/update_repository_references.py --cleanup .backup_20240101_120000

# JSON output format
python3 scripts/update_repository_references.py --format json --files file.txt
```

**Wrapper Script Usage:**
```bash
# Show help
./scripts/run_update.sh --help

# Dry run (no confirmation required)
./scripts/run_update.sh --dry-run --files docker-compose.yml README.md

# Update files (requires confirmation)
./scripts/run_update.sh --files docker-compose.yml scripts/README.md

# Rollback from backup
./scripts/run_update.sh --rollback .backup_20240101_120000

# Cleanup backup
./scripts/run_update.sh --cleanup .backup_20240101_120000
```

**Recommended Workflow:**
1. **Run audit** to identify files that need updating:
   ```bash
   ./scripts/run_audit.sh
   ```

2. **Dry run** to preview changes:
   ```bash
   ./scripts/run_update.sh --dry-run --files file1.txt file2.py
   ```

3. **Update files** (backups created automatically):
   ```bash
   ./scripts/run_update.sh --files file1.txt file2.py
   ```

4. **Verify changes** are correct:
   ```bash
   git diff
   ```

5. **Either cleanup or rollback**:
   ```bash
   # If changes are good, cleanup backup
   ./scripts/run_update.sh --cleanup .backup_20240101_120000
   
   # If changes need to be reverted, rollback
   ./scripts/run_update.sh --rollback .backup_20240101_120000
   ```

**Validation:**
The script automatically validates syntax for supported file types:
- **Python (.py)**: Compiles code to check for syntax errors
- **JSON (.json)**: Validates JSON structure
- **YAML (.yml, .yaml)**: Validates YAML structure (requires PyYAML)
- **Shell (.sh)**: Validates using `bash -n`
- **Markdown (.md)**: Checks for balanced code blocks

If validation fails, the file is not updated and the error is reported.

**Backup Structure:**
Backups are stored in `.backup_<timestamp>/` directory with the same directory structure as the original files:
```
.backup_20240101_120000/
├── docker-compose.yml
├── README.md
└── scripts/
    └── README.md
```

**Output Format (Text):**
```
================================================================================
REPOSITORY REFERENCE UPDATE REPORT
================================================================================

Old Name: old-repo-name
New Name: new-repo-name
Root Directory: /path/to/repo
Backup Directory: /path/to/repo/.backup_20240101_120000
Dry Run: False

Files Processed: 3
Files Updated: 3
Files Failed: 0
Total Occurrences Replaced: 15

UPDATED FILES:
--------------------------------------------------------------------------------
  docker-compose.yml
    Occurrences: 5
    Validation: YAML syntax valid
    Backup: .backup_20240101_120000/docker-compose.yml

  README.md
    Occurrences: 8
    Validation: Markdown structure valid
    Backup: .backup_20240101_120000/README.md

  scripts/README.md
    Occurrences: 2
    Validation: Markdown structure valid
    Backup: .backup_20240101_120000/scripts/README.md

================================================================================
```

**Error Handling:**
- **Read-only files**: Skipped with warning
- **Binary files**: Should be excluded from file list
- **Syntax errors**: File not updated, error reported
- **Backup failures**: Update aborted for that file
- **Validation failures**: File not updated, original preserved

**Requirements Satisfied:**
- **Requirement 2.2:** Update Docker service names
- **Requirement 2.3:** Update hardcoded container names in scripts
- **Requirement 2.4:** Replace old name in configuration files

---

## Database Backup Scripts

## Backup Script

### Location
`scripts/backup_database.sh`

### Purpose
Creates a timestamped backup of the WhizDB PostgreSQL database before performing migrations or other potentially destructive operations.

### Usage

**From host machine (recommended):**
```bash
./scripts/backup_database.sh
```

**From inside Docker container:**
```bash
docker-compose exec whiz-agent bash /app/scripts/backup_database.sh
```

### Backup Location
Backups are stored in `./backups/` directory with the naming format:
```
whizdb_backup_YYYYMMDD_HHMMSS.sql
```

Example: `whizdb_backup_20250119_143022.sql`

### Features
- ✅ Automatic timestamp generation
- ✅ Creates backup directory if it doesn't exist
- ✅ Works both on host and inside Docker containers
- ✅ Automatically detects credentials from the running container
- ✅ Includes `--clean` and `--if-exists` flags for safe restoration
- ✅ Displays backup size and location
- ✅ Shows restore command for convenience
- ✅ Lists recent backups

### Backup Contents
The backup includes:
- All table schemas
- All data from existing tables
- Indexes
- Constraints
- Views

The backup excludes:
- Ownership information (uses `--no-owner`)
- Privilege grants (uses `--no-privileges`)

This makes the backup portable across different PostgreSQL installations.

### Restoring a Backup

**From host machine:**
```bash
docker exec -i whiz-database psql -U admin -d WhizDB < backups/whizdb_backup_20250119_143022.sql
```

**From inside container:**
```bash
PGPASSWORD=whiz psql -h localhost -p 5432 -U admin -d WhizDB -f /app/backups/whizdb_backup_20250119_143022.sql
```

### Before Migration Workflow

1. **Backup the database:**
   ```bash
   ./scripts/backup_database.sh
   ```

2. **Verify backup was created:**
   ```bash
   ls -lh backups/
   ```

3. **Run migration:**
   ```bash
   docker exec whiz-database psql -U admin -d WhizDB -f /docker-entrypoint-initdb.d/migrate_v2.sql
   ```

4. **If migration fails, restore backup:**
   ```bash
   docker exec -i whiz-database psql -U admin -d WhizDB < backups/whizdb_backup_YYYYMMDD_HHMMSS.sql
   ```

### Requirements Satisfied
- **Requirement 8.2**: Backup existing data before migration

### Troubleshooting

**Error: PostgreSQL container is not running**
```bash
docker-compose up -d postgres
```

**Error: Permission denied**
```bash
chmod +x scripts/backup_database.sh
```

**Error: Directory not found**
The script automatically creates the `backups/` directory, but ensure you're running from the project root.

### Backup Retention
Backups are not automatically deleted. To manage disk space:

**List all backups:**
```bash
ls -lh backups/
```

**Delete old backups (older than 30 days):**
```bash
find backups/ -name "whizdb_backup_*.sql" -mtime +30 -delete
```

**Keep only the 5 most recent backups:**
```bash
ls -t backups/whizdb_backup_*.sql | tail -n +6 | xargs rm -f
```


## Refund Guide Processing Script

### Location
`scripts/process_refund_guide.py`

### Purpose
Automates the conversion of raw refund guide text files into structured JSON for use by the Parlant AI agent's policy-based decision making system.

### Usage

```bash
python scripts/process_refund_guide.py
```

### Input Files
- `parlant/context/raw/ops_refund_guide_1_10.txt`
- `parlant/context/raw/ops_refund_guide_11_23.txt`

### Output Files
- `parlant/context/processed/refund_guide.json` - Structured policy guidance for LLM context

### Features
- Reads all `.txt` files in the raw directory
- Cleans text (removes page breaks, fixes OCR errors)
- Extracts section titles and content
- Generates structured JSON with proper formatting
- Overwrites existing processed file

**Note:** Business rules are hardcoded in `parlant/tools/rule_engine.py`, not extracted from this guide. This JSON file provides context for LLM analysis only.

### When to Run
Run this script whenever the operations refund guide is updated:
- New policy changes
- Updated procedures
- Additional refund scenarios
- Rule modifications

### Example Output

```
=== Refund Guide Processing Script ===

Found 2 raw file(s):
  - ops_refund_guide_1_10.txt
  - ops_refund_guide_11_23.txt

Processing refund guide...
  Extracted 15 sections:
    - Pre-Arrival
    - Oversold
    - No Attendant
    - Missing Amenity
    - Poor Experience
    ...

✓ Saved parlant/context/processed/refund_guide.json
  Size: 45,231 bytes

✓ Processing complete!
```

### After Running
After processing the guide, restart the Parlant service to reload the updated policies:

```bash
docker-compose restart parlant
```
