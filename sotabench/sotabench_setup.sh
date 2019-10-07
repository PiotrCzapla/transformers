#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
REPO="$( cd "$(dirname "$0")" ; pwd -P )"

$PYTHON -m pip install -e .
$PYTHON -m pip install torch
$PYTHON -m pip install ftfy==4.4.3
$PYTHON -m pip install spacy
$PYTHON -m spacy download en

echo Running temporary setup scripts
apt-get install -y git
cd /workspace
git clone https://github.com/PiotrCzapla/sotabench-eval.git
cd sotabench-eval
git pull
$PYTHON -m pip install -e .