# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForQuestionAnswering, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering,
                          XLMTokenizer, XLNetConfig,
                          XLNetForQuestionAnswering,
                          XLNetTokenizer,
                          DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from .utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

SQUAD_PRETRAINED='bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking-finetuned-squad'

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer,  dataset, examples, features, one_batch=False, prefix="", run_eval=False):
    if not os.path.exists(args.output_dir) :
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            # XLM & DistilBert don't use segment_ids
            if args.model_type not in ['xlm', 'distilbert']:
                inputs['token_type_ids'] = batch[2]
            
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)
    
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(
                                               outputs[0][i]),
                                           start_top_index=to_list(
                                               outputs[1][i]),
                                           end_top_log_probs=to_list(
                                               outputs[2][i]),
                                           end_top_index=to_list(
                                               outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
         
            all_results.append(result)
        if one_batch:
            #fill all results with dummy predictions so that write_predictions work
            predicted_examples = set(r.unique_id for r in all_results)
            dummy = all_results[-1]
            all_results += [dummy._replace(unique_id=f.unique_id) for f in features if f.unique_id not in predicted_examples]
            break

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                                   args.max_answer_length, output_prediction_file,
                                   output_nbest_file, output_null_log_odds_file, args.predict_file,
                                   model.config.start_n_top, model.config.end_n_top,
                                   args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                          args.max_answer_length, args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)

    return output_prediction_file

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    
    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor(
        [f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor(
            [f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def collect_answers(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
    

def run_evaluation(model_name, pretrained_weights, model_type='bert', cased=False, 
                   paper_arxiv_id='1810.04805', paper_pwc_id='bert-pre-training-of-deep-bidirectional', 
                   cache={}):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default=None, type=str, required=False,
    #                     help="Which models should we evaluate on")
    # parser.add_argument("--force_full_run",  action='store_true',
    #                     help='Ensures that we run a full model even if the cache matches the first batch.')
    # args = parser.parse_args()
    
    args = argparse.Namespace()
    args.model_name = pretrained_weights
    args.do_lower_case = not cased

    args.force_full_run = False
    args.doc_stride = 128
    args.fp16 = False
    args.fp16_opt_level = 'O1'
    args.local_rank = -1,
    args.max_answer_length = 30
    args.max_query_length = 64
    args.max_seq_length = 384
    args.model_type = model_type
    args.n_best_size = 20
    args.no_cuda = False
    args.null_score_diff_threshold = 0.0
    
    args.output_dir = '/tmp/squad'
    
    args.overwrite_cache = True
    args.per_gpu_eval_batch_size = 12
    args.seed = 0
    args.verbose_logging = True
    args.version_2_with_negative = False

    evaluator = SQuADEvaluator(
        local_root=Path.home()/".cache/sotabench/data/squad/",
        model_name=model_name,
        version=SQuADVersion.V11,
        paper_arxiv_id=paper_arxiv_id,
        paper_pwc_id=paper_pwc_id,
    )
    
    args.predict_file = evaluator.dataset_path

    args.output_dir = "/tmp/squad"
    args.device = "cuda"
    if args.no_cuda:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
       
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.verbose_logging else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, args.fp16)

    set_seed(args)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    model = model_class.from_pretrained(args.model_name, config = config)

    model.to(args.device)
    if model_type not in cache:
        cache[model_type]  = load_and_cache_examples(
            args, tokenizer, evaluate=True, output_examples=True)
    
    dataset, examples, features = cache[model_type]
    file_path = evaluate(args, model, tokenizer,  dataset, examples, features, one_batch=True, prefix='one_batch', run_eval=False)
    evaluator.add(collect_answers(file_path))
    if not evaluator.cache_exists:
        logger.info("Cache not found resetting and run on the full dataset")
        args.force_full_run = True

    if args.force_full_run:
        logger.info("Reset evaluator and rerun on the full dataset")
        evaluator.reset()
        file_path = evaluate(args, model, tokenizer, dataset, examples, features,
                    one_batch=False, prefix='predictions_full')
        evaluator.add(collect_answers(file_path))

    results = evaluator.save()
    logger.info("Accuracy on full dataset: " + repr(evaluator.results))

def main():
    run_evaluation('BERT large (whole word masking, uncased)',
                    'bert-large-uncased-whole-word-masking-finetuned-squad')
    run_evaluation('BERT large (whole word masking, cased)',
                   'bert-large-uncased-whole-word-masking-finetuned-squad',
                    cased=True)
    run_evaluation('DistilBERT',
                    'distilbert-base-uncased-distilled-squad',
                    model_type="distilbert",
                    paper_pwc_id='distilbert-a-distilled-version-of-bert',
                    paper_arxiv_id='1910.01108')

if __name__ == "__main__":
    main()
