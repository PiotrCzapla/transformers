#%%
from sotabencheval.language_modeling.wikitext import WikiText103Eval, perplexity_evauluate
import sys
import numpy as np
import re
from itertools import islice
from functools import reduce
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm
import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, TransfoXLCorpus

from sotabencheval.language_modeling.wikitext import WikiText103Eval, perplexity_evauluate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)
logger = logging.getLogger(__name__)

def iterate_over_batches(data, bs, bptt):        
    def batched(Z, bptt):
        sz = Z.shape[-1]
        for s in range(0, sz, bptt):
            yield Z[..., s:s+bptt]
    size = data.numel()
    batched_size = ((size-1) // bs) * bs
    # filp - to be able to switch to batch_size 1 later and maintain trasnfoxl memory
    X = data[:batched_size].view(bs, -1).flip(0,)
    Y = data[1:batched_size+1].view(bs, -1).flip(0,)
    yield from zip(batched(X, bptt), batched(Y, bptt))
    X = data[None, batched_size:-1]
    Y = data[None, batched_size+1:]
    yield from zip(batched(X, bptt), batched(Y, bptt))

def read_wiki():
    with open('data/wikitext-103/wiki.test.tokens') as f:
        return f.readlines()

def evaluate(experiemnt, log_probs_generator, model, test_data, batch_size=8, bptt=128, device='cuda'):
    test_data = torch.tensor(test_data).to(device)
    model = model.to(device)
    data_iter = iterate_over_batches(test_data, bs=batch_size, bptt=bptt)
    model.eval()
    with torch.no_grad():
        total_steps = len(test_data)//(batch_size*bptt)
        log_probs_generator_instance = log_probs_generator(model, data_iter)
        for log_probs, y in tqdm(log_probs_generator_instance, total = total_steps):
            experiemnt.add(log_probs, y)
            #TODO batch caching
        results = experiemnt.save()
        print("Evaluation results:", results.to_dict()["results"])
        
## Transformers XL

def setup_transfo_xl(model_name):
    def _fix_tokenizer_encoding(tokenizer):
        import collections
        if '–' not in tokenizer.sym2idx:
            tokenizer.idx2sym = [sym.encode('latin1').decode(
                'utf-8') for sym in tokenizer.idx2sym]
            tokenizer.sym2idx = collections.OrderedDict((sym.encode('latin1').decode('utf-8'), idx)
                                                        for sym, idx in tokenizer.sym2idx.items())
        else:
            logger.info("No need to fix tokenizer encoding")
        return tokenizer

    model = TransfoXLLMHeadModel.from_pretrained(model_name)
    tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
    tokenizer = _fix_tokenizer_encoding(tokenizer)

    def encode(lines):
        # TODO: tokenize is removing the empty lines and add_eos is not being added.
        # TODO2: tokenize in transformers xl does not handle multiple lines correctly (removes <eos>)
        return tokenizer.convert_tokens_to_ids(
            [tok for l in lines for tok in tokenizer._tokenize(l.strip(), add_eos=True)])
    tokenizer.encode = encode
    
    return model, tokenizer

def evaluate_transfo_xl(wikitext103_testset):
    experiment = WikiText103Eval(
        model_name="transfo-xl",
        paper_arxiv_id="1901.02860",
        paper_pwc_id="transformer-xl-attentive-language-models",
        model_description="English model trained on wikitext-103: 18-layer, 1024-hidden, 16-heads, 257M parameters.",
        #expected perplexity: 18.3, our: 18.19  (due to fixes in the batching)
    )

    model, tokenizer = setup_transfo_xl("transfo-xl-wt103")
    test_data = tokenizer.encode(wikitext103_testset)
    assert len(test_data) == 245569
    logger.info("Text tokenized to %d tokens. according to paper it should be 245569", len(test_data))
    
    def log_probs_generator(model, data_iter):
        past = {}
        for x, y in data_iter:
            log_probs, mems, *_ = model(input_ids=x, **past)
            past = {'mems': mems}
            yield log_probs, y
    
    evaluate(experiment, log_probs_generator, model, test_data,  batch_size=8, bptt=128)

## GPT2 evaulation

def setup_gpt2(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def fix_moses(line, _md=MosesDetokenizer(lang='en')):
    return _md.detokenize(line.split(), unescape=False) + "\n"

def fix_header(line, _header_re=re.compile(r'^(=( =)*)( [^=]+ )\1$')):
    match = _header_re.match(line)
    if match:
        markup = match.group(1).replace(" ", "")
        return markup+match.group(3)+markup
    return line

def fix_unk(replacement):
    def fix(line):
        return line.replace("<unk>", replacement)
    fix.__name__ = f"fix_unk_{replacement}"
    return fix

def preprocess_text(lines, fixes=[]):
    return "".join(reduce(lambda v, f: f(v), fixes, l) for l in lines)

def evaluate_gpt2_small(wikitext103_testset):
    experiment = WikiText103Eval(
        model_name="GPT2-small",
        paper_pwc_id="language-models-are-unsupervised-multitask",
        model_description="OpenAI GPT2-small: 12-layer, 768-hidden, 12-heads, 117M parameters.",
        text_transformation = True,
        subword_tokenization = True,
        #expected perplexity: 37.50, our: 36.49 ( or better if we predict 1 word at a time )
    )

    model, tokenizer = setup_gpt2("gpt2")
    
    fixes = [fix_moses, fix_header, fix_unk('[unknown]')]
    tokenizer.max_len = 2**62
    test_data = tokenizer.encode(preprocess_text(wikitext103_testset, fixes))
    
    def log_probs_generator(model, data_iter):
        for x, y in data_iter:
            logits, *_ = model(input_ids=x)
            yield torch.log_softmax(logits, dim=-1), y
    
    evaluate(experiment, log_probs_generator, model, test_data,  batch_size=8, bptt=1024)

## GPT1

def setup_gpt(model_name="openai-gpt"):
    model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    return model, tokenizer

def evaluate_gpt(wikitext103_testset):
    experiment = WikiText103Eval(
        model_name="GPT",
        paper_pwc_id="",
        model_description="OpenAI GPT: 12-layer, 768-hidden, 12-heads, 110M parameters.",
        text_transformation=True,
        subword_tokenization = True,
    )

    model, tokenizer = setup_gpt("openai-gpt")

    # fixes = [fix_moses, fix_header, fix_unk('[unknown]')]
    # test_data = tokenizer.encode(preprocess_text(wikitext103_testset, fixes))

    # def log_probs_generator(model, data_iter):
    #     for x, y in data_iter:
    #         logits, *_ = model(input_ids=x)
    #         yield torch.log_softmax(logits), y

    # evaluate(experiment, log_probs_generator,
    #          test_data,  batch_size=8, bptt=128)

## XLNet
def setup_xlnet(model_name):
    model = XLNetLMHeadModel.from_pretrained(model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    return model, tokenizer


def evaluate_xlnet_base(wikitext103_testset):
    experiment = WikiText103Eval(
        model_name="XLNet-base-cased",
        paper_arxiv_id="...",
        paper_pwc_id="...",
        model_description="12-layer, 768-hidden, 12-heads, 110M parameters.",
        text_transformation=True,
        subword_tokenization = True,
      
    )

#xlnet-large-cased
evaluators, evaluator_names = list(zip(*[(v, n.replace("evaluate_", ""))
                                    for n, v in globals().items() if n.startswith('evaluate_')]))

def main():
    import subprocess, os
    subprocess.run("bash ./sotabench-setup.sh",  shell=True,
                   cwd=os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=False,
                        help="Model to evaulate %s" % str(evaluator_names))
    args = parser.parse_args()
    wikitext103_testset = read_wiki()
    for evaluator in evaluators:
        if (args.model or "") in evaluator.__name__:
            print("Running", evaluator.__name__)
            evaluator(wikitext103_testset)

if __name__ == '__main__':
    main()


#%%
