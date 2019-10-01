#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}

apt-get install -y git

echo Running temporary setup scripts
cd ~
SOTABENCH=~/.cache/sotabench
mkdir -p $SOTABENCH
cd $SOTABENCH
git clone https://github.com/PiotrCzapla/sotabench-eval.git
cd sotabench-eval
git pull
pip install -e .
pip install torch

apt-get install -y unzip
mkdir -p $SOTABENCH/data
cd $SOTABENCH/data
ls
find /.data
[ -d wikitext-103 ] || unzip  /.data/nlp/wiki-text-103/wikitext-103-v1.zip
[ -d wikitext-2 ] || unzip  /.data/nlp/wiki-text-103/wikitext-103-v1.zip