echo Running temporary setup scripts
cd ~
mkdir -p sotabench
cd ~/sotabench
git clone https://github.com/PiotrCzapla/sotabench-eval.git
cd sotabench-eval
git pull
pip install -e .

