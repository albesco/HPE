#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${1:-vitpose_project}"

echo "Creating project structure in: ${PROJECT_ROOT}"

DIRECTORIES=(
  "script"
  "src/vitpose_base"
  "src/hpe_project/detection"
  "src/hpe_project/pose"
  "src/hpe_project/pipeline"
  "src/hpe_project/utils"
  "configs/detection"
  "configs/pose"
  "configs/pipeline"
  "data/dataset"
  "data/input"
  "data/intermediate"
  "data/output"
  "models/detection"
  "models/pose"
  "outputs"
  "logs"
  "notebooks"
)

for dir_path in "${DIRECTORIES[@]}"; do
  mkdir -p "${PROJECT_ROOT}/${dir_path}"
done

touch "${PROJECT_ROOT}/README.md"
touch "${PROJECT_ROOT}/requirements.txt"

cat > "${PROJECT_ROOT}/.gitignore" <<'EOF'
venv/
data/
models/
outputs/
logs/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/
EOF

echo "Done."
echo "Next steps:"
echo "1. cd ${PROJECT_ROOT}"
echo "2. python3 -m venv venv"
echo "3. source venv/bin/activate"
echo "4. git init"
