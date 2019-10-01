echo Running temporary setup scripts
cd ~
SOTABENCH=~/.cache/sotabench
mkdir -p $SOTABENCH
cd $SOTABENCH
git clone https://github.com/PiotrCzapla/sotabench-eval.git
cd sotabench-eval
git pull
pip install -e .

cd $SOTABENCH
mkdir -p data
cd data
[ -f wikitext-103-v1.zip ] || wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
[ -f wikitext-2-v1.zip ] || wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip

[ -d wikitext-2 ] || unzip wikitext-2-v1.zip
[ -d wikitext-103 ] || unzip wikitext-103-v1.zip