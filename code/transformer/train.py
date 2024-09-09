import torch
import torch.nn as nn
import time
import re
import sys
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
import json
import os
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower
from qtorch.quant import *
from qtorch.optim import OptimLP
import logging
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()



os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import configs
from datasets import TranslateDataset
from models import Transformer



def read_data(source_file, target_file):
    source_data = open(source_file).read().strip().split("\n")
    target_data = open(target_file).read().strip().split("\n")
    return source_data, target_data


def validate_epoch(model, valid_loader, epoch, n_epochs, source_pad_id, target_pad_id, device):
    model.eval()
    total_loss = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        preds = model(source, target_input, source_mask, target_mask)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])

    valid_loss = sum(total_loss) / len(total_loss)
    return valid_loss, total_loss


def train_epoch(times, model, train_loader, optim, epoch, n_epochs, source_pad_id, target_pad_id, device):
    model.train()
    total_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        
        optim.zero_grad()
        gold = target[:, 1:].contiguous().view(-1)
        for i in range(times):  
            preds = model(source, target_input, source_mask, target_mask)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
            loss.backward()
        for param in model.parameters():
            param.grad.data /= times
        optim.step()
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])
    
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss, total_loss

def load_model_tokenizer(configs):
    """
    This function will load model and tokenizer from pretrained model and tokenizer
    """

    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])    
    return source_tokenizer, target_tokenizer

# def translate(model, sentence, source_tokenizer, target_tokenizer, source_max_seq_len=256, 
#     target_max_seq_len=256, beam_size=3, device=torch.device("cpu"), print_process=False):
def translate(model, sentence, source_tokenizer, target_tokenizer, source_max_seq_len=256, 
    target_max_seq_len=256, beam_size=3, device=configs['device'], print_process=False):
    """
    This funciton will translate give a source sentence and return target sentence using beam search
    """
    # Convert source sentence to tensor
    source_tokens = source_tokenizer.encode(sentence)[:source_max_seq_len]
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, source_tokenizer.pad_token_id).to(device)
    # Feed forward Encoder
    encoder_output = model.encoder.forward(source_tensor, source_mask)
    # Initialize beam list
    beams = [([target_tokenizer.bos_token_id], 0)]
    completed = []
    # Start decoding
    for _ in range(target_max_seq_len):
        new_beams = []
        for beam in beams:
            # Get input token
            input_token = torch.tensor([beam[0]]).to(device)
            # Create mask
            target_mask = model.make_target_mask(input_token).to(device)
            # Decoder forward pass
            pred = model.decoder.forward(input_token, encoder_output, source_mask, target_mask)
            # Forward to linear classify token in vocab and Softmax
            pred = F.softmax(model.final_linear(pred), dim=-1)
            # Get tail predict token
            pred = pred[:, -1, :].view(-1)
            # Get top k tokens
            top_k_scores, top_k_tokens = pred.topk(beam_size)
            # Update beams
            for i in range(beam_size):
                new_beams.append((beam[0] + [top_k_tokens[i].item()], beam[1] + top_k_scores[i].item()))
        
        import copy
        beams = copy.deepcopy(new_beams)
        # Sort beams by score
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # Add completed beams to completed list and reduce beam size
        for beam in beams:
            if beam[0][-1] == target_tokenizer.eos_token_id:
                completed.append(beam)
                beams.remove(beam)
                beam_size -= 1
        
        # Print screen progress
        if print_process:
            print(f"Step {_+1}/{target_max_seq_len}")
            print(f"Beam size: {beam_size}")
            print(f"Beams: {[target_tokenizer.decode(beam[0]) for beam in beams]}")
            print(f"Completed beams: {[target_tokenizer.decode(beam[0]) for beam in completed]}")
            print(f"Beams score: {[beam[1] for beam in beams]}")
            print("-"*100)

        if beam_size == 0:
            break


    # Sort the completed beams
    completed.sort(key=lambda x: x[1], reverse=True)
    # Get target sentence tokens
    # if completed = []:
    #     print()
        # return target_tokenizer.decode(target_tokens, skip_special_tokens=True)
    if completed == []:
        target_tokens = []
    else:
        target_tokens = completed[0][0]
    # Convert target sentence from tokens to string
    target_sentence = target_tokenizer.decode(target_tokens, skip_special_tokens=True)
    return target_sentence
    

def calculate_bleu_score(model, source_tokenizer, target_tokenizer, configs):
    # device = torch.device("cpu")
    device = configs['device']
    model.eval()
    model.to(device)
    def preprocess_seq(seq):
        seq = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq
    
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    pred_sents = []
    for sentence in tqdm(valid_src_data):
        pred_trg = translate(model, sentence, source_tokenizer, target_tokenizer, configs["source_max_seq_len"], configs["target_max_seq_len"], configs["beam_size"], device)
        pred_sents.append(pred_trg)

    hypotheses = [preprocess_seq(sent).split() for sent in pred_sents]
    references = [[sent.split()] for sent in valid_trg_data]
    
    weights = [(0.5, 0.5),(0.333, 0.333, 0.334),(0.25, 0.25, 0.25, 0.25)]
    bleu_2 = corpus_bleu(references, hypotheses, weights=weights[0])
    bleu_3 = corpus_bleu(references, hypotheses, weights=weights[1])
    bleu_4 = corpus_bleu(references, hypotheses, weights=weights[2])
    print(f"BLEU-2: {bleu_2} | BLEU-3: {bleu_3} | BLEU-4: {bleu_4}")
    return {"bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}

def train(times, model, train_loader, valid_loader, optim, n_epochs, source_pad_id, target_pad_id, device, model_path, early_stopping):
    best_val_loss = np.Inf
    best_epoch = 1
    count_early_stop = 0
    for epoch in range(n_epochs):
        train_loss, train_losses = train_epoch(
            times,
            model=model,
            train_loader=train_loader,
            optim=optim,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )
        valid_loss, valid_losses = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch + 1
            # save model
            torch.save(model.state_dict(), model_path)
            print("---- Detect improment and save the best model ----")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if count_early_stop >= early_stopping:
                print("---- Early stopping ----")
                break


        



def main(bit, times):
    name = bit + '_' + times
    logging.basicConfig(filename='./result/' +name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    round_bit = int(bit)
    times = int(times)
    our_bit = BlockFloatingPoint(wl=round_bit, dim=0)
    # define quantization functions
    weight_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    grad_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    momentum_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    acc_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])

    model = Transformer(
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )
    model = sequential_lower(
        model,
        layer_types=["linear", "dropout"],  ## type of quantization
        forward_number=our_bit,
        backward_number=our_bit,
        forward_rounding="stochastic",
        backward_rounding="stochastic",
    )
    model.final_linear = model.final_linear[0]  # removing the final quantization module
    # print(model.final_linear)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)
    optim = OptimLP(optim,
                        weight_quant=weight_quant,
                        grad_quant=grad_quant,
                        momentum_quant=momentum_quant,
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"parameter number: {num_parameters}")
    for name, param in model.named_parameters():
        print(f"Name: {name}, ")
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=train_src_data, 
        target_data=train_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )
    valid_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=valid_src_data, 
        target_data=valid_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )

    device = torch.device(configs["device"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        shuffle=False
    )

    model.to(configs["device"])
    train(times, model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        n_epochs=configs["n_epochs"],
        source_pad_id=source_tokenizer.pad_token_id,
        target_pad_id=target_tokenizer.pad_token_id,
        device=device,
        model_path=configs["model_path"],
        early_stopping=configs["early_stopping"]
    )




if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
