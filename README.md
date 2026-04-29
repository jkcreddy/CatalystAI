
Steps to install python virtual environment

brew install uv
uv venv --python cpython-3.11.6-macos-aarch64-none
source .venv/bin/activate

uv pip install -r requirements.txt

export PYTHONPATH=$(pwd)