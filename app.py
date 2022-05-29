import flask
from flask_cors import CORS
from flask import request, request, jsonify, render_template
from collections import OrderedDict, Counter
import collections
from pprint import pprint
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import argparse
import random
import json
import re
import os
import pickle
from datetime import datetime
import nltk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import torch
import warnings
warnings.filterwarnings('ignore')


############ Hardware Environment ############
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# WANDB_INTEGRATION = True
# if WANDB_INTEGRATION:
#     import wandb
#     wandb.login()

# tokenization params
TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-large")
encoder_max_length = 256
decoder_max_length = 256

### Set Training Arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', default="/home/ubuntu/P2MCQ_umich/checkpoint-40000", type=str,
                    help="Pretrained model path.")

parser.add_argument('--run_sample', default=False, type=bool,
                    help="Run the hard-coded samples.")


def load_model(path: str = "/home/oliviaaa/ControllableParaphrase/Experiment/Baseline/BART-Concat/BART_CONCAT_OUT/checkpoint-40000"):
    """Load pretrained checkpoint."""
    return AutoModelForSeq2SeqLM.from_pretrained(
        path).to(DEVICE)


def generate_summary(test_samples, model, tokenizer=TOKENIZER):
    """Generate summary/paraphrase using model and test samples."""
    inputs = tokenizer(
        test_samples["src"],
        test_samples["overlap"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    # print('Device: ', model.device)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(
        input_ids, attention_mask=attention_mask, max_length=256)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


def get_controllable_para(src_sent, overlap, tgt_sent=None, verbose=False):
    if type(src_sent) == str:
        src_sent = [src_sent]
        overlap = [overlap]
    para_after_tuning = []
    cur_idx = 0
    # last_idx = min(len(src_sent), cur_idx+20)
    while cur_idx+20 < len(src_sent):

        test_sample = {
            'src': src_sent[cur_idx:cur_idx+20],
            'overlap': overlap[cur_idx:cur_idx+20],
        }
        para_after_tuning.extend(generate_summary(test_sample, MODEL)[1])
        cur_idx = cur_idx + 20

    test_sample = {
        'src': src_sent[cur_idx:],
        'overlap': overlap[cur_idx:],
    }
    para_after_tuning.extend(generate_summary(test_sample, MODEL)[1])

    para_after_tuning = [x.replace('\n', '') for x in para_after_tuning]

    # if verbose:
    all_hit = 0
    if tgt_sent is not None:
        for src, tgt, lap, pred in zip(src_sent, tgt_sent, overlap, para_after_tuning):
            if verbose:
                print("=======")
                print(f' Source:    {src}')
                print(f' Target:    {tgt}')
                print(f' Overlap:   {lap}')
                print(f'-> Pred:    {pred}')
                hit = (lap in pred)
                # if hit:
                #     hit_value = '1'
                print(f'==> Hit? {hit}')
                all_hit += 1
    else:
        for src, lap, pred in zip(src_sent, overlap, para_after_tuning):
            if verbose:
                print("=======")
                print(f' Source:    {src}')
                # print(f' Target:    {tgt}')
                print(f' Overlap:   {lap}')
                print(f'-> Pred:    {pred}')
                hit = (lap in pred)
                # if hit:
                #     hit_value = '1'
                print(f'==> Hit? {hit}')
                all_hit += 1
    hit_rate = float(all_hit)/float(len(src_sent))
    if verbose:
        print(f'===== Overall Hit-rate: {hit_rate*100}% =====')
    return para_after_tuning, hit_rate


### Hard-coded Samples ###
repeated_src_sents = [
    'we use the stanford dependency parser to extract nouns and their grammatical roles .']*3
repeated_tgt_sents = [
    'we extract lexical relations from the question using the stanford dependencies parser .']*3
overlappings = [
    'stanford dependency parser',
    'extract nouns',
    'grammatical roles',
]


def web_test(input_sent, overlap, model):
    test_sample = {
        'src': input_sent,
        'overlap': overlap,
    }
    res = generate_summary(test_sample, model, tokenizer=TOKENIZER)
    # print('Web test => ', res)
    return res[1][0].replace('\n', '')


app = flask.Flask(__name__)
app.config['DEBUG'] = True

CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    if args.run_sample:
        print("========= Sample Running =========")
        hard_coded_sample_para, hard_coded_sample_hitrate = get_controllable_para(
            repeated_src_sents, overlappings, verbose=True)
        return f"""
        <h1> Predicting! </h1>
        <div> <p>{hard_coded_sample_para}</p> </div>"""
    else:
        input_sent = request.args['src_sent']
        overlap = request.args['span']
        paragraph = request.args['paragraph']
        func = request.args['func']
        # print('Func: ', func)
        if func != "0":
            return f"""<div>{func} has not be deployed!</div>"""
        result = web_test(input_sent, overlap, model=MODEL)
        # print('Result: ', result)
        return f"""{result}"""


if __name__ == '__main__':
    args = parser.parse_args()
    MODEL = load_model(args.model_ckpt)
    app.run(debug=True)
