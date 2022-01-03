import enum
import pandas as pd
import numpy as np
import torch
import argparse
import os,json
import sys
from tqdm import tqdm
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from torch.nn.modules import loss
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import transformers
import random
from preprocess import create_dataloader, get_conll_data
from utils import compute_loss, get_ent_tags, batch_to_device, compute_f1
from models import NERNetwork
from typing import List
from visdom import Visdom
import logging

save_dir='/home/xhsun/Desktop/NER_Parsing/train_models/baseline_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)

logger=logging.getLogger('main')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# System based
random.seed(seed)
np.random.seed(seed)

vis=Visdom(env='main',log_to_filename=os.path.join(save_dir,'vis_log'))

device='cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device {}".format(device))

def train(args,
        train_dataloader,
        tag_encoder,
        train_conll_tags,
        dev_conll_tags,
        dev_dataloader):
    
    n_tags=tag_encoder.classes_.shape[0]
    logger.info("n_tags : {}".format(n_tags))

    print_loss_step=len(train_dataloader)//5
    evaluation_steps=len(train_dataloader)//2
    logger.info("Under an epoch, loss will be output every {} step, and the model will be evaluated every {} step".format(print_loss_step,evaluation_steps))


    model=NERNetwork(model_name_or_path=args.bert_model_name_or_path,n_tags=n_tags)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
        logger.info("Load ckpt to continue training !")
    model.to(device=device)
    logger.info("Using device : {}".format(device))
    optimizer_parameters=model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = args.learning_rate)
    num_train_steps=int(len(train_conll_tags)//args.train_batch_size)*args.epochs
    warmup_steps=int(num_train_steps*args.warmup_proportion)
    logger.info("num_train_steps : {}, warmup_proportion : {}, warmup_steps : {}".format(num_train_steps,args.warmup_proportion,warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )

    global_step=0
    
    previous_f1=-1
    predictions=predict(model=model,test_dataloader=dev_dataloader,tag_encoder=tag_encoder,device=device)
    f1=compute_f1(pred_tags=predictions,golden_tags=dev_conll_tags)
    if f1>previous_f1:
        logger.info("Previous f1 score is {} and current f1 score is {}".format(previous_f1,f1))
        previous_f1=f1

    for epoch in range(args.epochs):
        model.train()
        model.zero_grad()
        training_loss=0.0
        for iteration, batch in tqdm(enumerate(train_dataloader)):
            batch=batch_to_device(inputs=batch,device=device)
            input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
            outputs=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)#(batch_size,seq_length,num_classes)
            loss=compute_loss(preds=outputs,target_tags=batch['target_tags'],masks=batch['attention_mask'],device=device,n_tags=n_tags)
            training_loss+=loss.item()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step+=1

            if iteration % print_loss_step == 0:
                training_loss/=print_loss_step
                logger.info("Epoch : {}, global_step : {}/{}, loss_value : {} ".format(epoch,global_step,num_train_steps,training_loss))
                training_loss=0.0

            vis.line([optimizer.param_groups[0]['lr']],[global_step],win=f'learning rate',
                    opts=dict(title=f'learning rate', xlabel='step', ylabel='lr'), update='append')
            vis.line([loss.item()],[global_step],win=f'training loss',
                    opts=dict(title=f'training loss', xlabel='step', ylabel='loss'), update='append')

            if iteration % evaluation_steps == 0 :
                predictions=predict(model=model,test_dataloader=dev_dataloader,tag_encoder=tag_encoder,device=device)
                f1=compute_f1(pred_tags=predictions,golden_tags=dev_conll_tags)
                if f1>previous_f1:
                    torch.save(model.state_dict(),f=os.path.join(save_dir,'pytorch_model.bin'))
                    logger.info("Previous f1 score is {} and current f1 score is {}, best model has been saved in {}".format(previous_f1,f1,os.path.join(save_dir,'pytorch_model.bin')))
                    previous_f1=f1                    
                    
                else:
                    args.patience-=1
                    logger.info("Left patience is {}".format(args.patience))
                    if args.patience==0:
                        logger.info("Total patience is {}, run our of patience, early stop!".format(args.patience))
                        return
                vis.line([f1],[global_step],win=f'F1-score',
                        opts=dict(title=f'F1-score', xlabel='step', ylabel='F1-score', markers=True, markersize=10), update='append')

                model.zero_grad()
                model.train()




def predict(model,test_dataloader,tag_encoder,device):
    logger.info("Evaluating the model...")
    if model.training:
        model.eval()
    
    predictions=[]
    for batch in test_dataloader:
        batch=batch_to_device(inputs=batch,device=device)
        input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        with torch.no_grad():
            outputs=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)#(batch_size,seq_length,num_classes)

        for i in range(outputs.shape[0]):
            indices = torch.argmax(outputs[i],dim=1)#(seq_length,)
            preds = tag_encoder.inverse_transform(indices.cpu().numpy())#(seq_length,)
            preds = [prediction for prediction, offset in zip(preds.tolist(), batch.get('offsets')[i]) if offset]
            preds = preds[1:-1]

            predictions.append(preds)
    
    return predictions


def main():
    parser = argparse.ArgumentParser()
    # input and output parameters
    parser.add_argument('--bert_model_name_or_path', default = '/home/xhsun/NLP/huggingfaceModels/English/bert-base-uncased', help='path to the BERT')
    parser.add_argument('--file_path', required=True, help='path to the ner data')
    parser.add_argument('--save_dir', default=save_dir, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None,help='Fine tuned model')
    # training parameters
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--dev_batch_size', default=64, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--max_len', default=196, type = int)
    parser.add_argument('--patience', default=10, type = int)

    #Other parameters
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--num_workers', default=1, type = int)
    args = parser.parse_args()

    train_conll_data=get_conll_data(split='train',dir=args.file_path)
    dev_conll_data=get_conll_data(split='valid',dir=args.file_path)
    test_conll_data=get_conll_data(split='test',dir=args.file_path)
    tag_scheme=get_ent_tags(all_tags=train_conll_data.get('tags'))
    tag_outside='O'
    if tag_outside in tag_scheme:
        del tag_scheme[tag_scheme.index(tag_outside)]
    tag_complete=[tag_outside]+tag_scheme
    with open(os.path.join(save_dir,'label.json'),'w') as f:
        json.dump(obj=' '.join(tag_complete),fp=f)
    logger.info("Tag scheme : {}".format(' '.join(tag_scheme)))
    logger.info("Tag has been saved in {}".format(os.path.join(save_dir,'label.json')))
    tag_encoder=sklearn.preprocessing.LabelEncoder()
    tag_encoder.fit(tag_complete)

    tokenizer_parameters=json.load(open(os.path.join(args.bert_model_name_or_path,'tokenizer_config.json')))
    transformer_tokenizer=AutoTokenizer.from_pretrained(args.bert_model_name_or_path,**tokenizer_parameters)
    transformer_config=AutoConfig.from_pretrained(args.bert_model_name_or_path)

    train_dataloader=create_dataloader(sentences=train_conll_data.get('sentences'),
                                      tags=train_conll_data.get('tags'),
                                      transformer_tokenizer=transformer_tokenizer,
                                      transformer_config=transformer_config,
                                      max_len=args.max_len,
                                      tag_encoder=tag_encoder,
                                      tag_outside=tag_outside,
                                      batch_size=args.train_batch_size,
                                      num_workers=args.num_workers)
    dev_dataloader=create_dataloader(sentences=dev_conll_data.get('sentences'),
                                      tags=dev_conll_data.get('tags'),
                                      transformer_tokenizer=transformer_tokenizer,
                                      transformer_config=transformer_config,
                                      max_len=args.max_len,
                                      tag_encoder=tag_encoder,
                                      tag_outside=tag_outside,
                                      batch_size=args.dev_batch_size,
                                      num_workers=args.num_workers)

    # args display
    args_config={}
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))
        args_config[k]=v

    with open(os.path.join(save_dir,'args_config.dict'),'w') as f:
        json.dump(args_config,f,ensure_ascii=False)

    train(args=args,train_dataloader=train_dataloader,
            tag_encoder=tag_encoder,
            train_conll_tags=train_conll_data.get('tags'),
            dev_conll_tags=dev_conll_data.get('tags'),
            dev_dataloader=dev_dataloader)


if __name__=="__main__":
    main()