from torch._C import device
from models import NERNetwork
from preprocess import create_dataloader, get_conll_data
import json,os
import torch
import sklearn.preprocessing
from transformers import AutoTokenizer, AutoConfig
from utils import batch_to_device,compute_f1

device='cuda' if torch.cuda.is_available() else 'cpu'

def predict(model,test_dataloader,tag_encoder,device):
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


save_dir='/home/xhsun/Desktop/NER_Parsing/train_models/baseline_models'

args=json.load(open(os.path.join(save_dir,'args_config.dict')))

tokenizer_parameters=json.load(open(os.path.join(args['bert_model_name_or_path'],'tokenizer_config.json')))
transformer_tokenizer=AutoTokenizer.from_pretrained(args['bert_model_name_or_path'],**tokenizer_parameters)
transformer_config=AutoConfig.from_pretrained(args['bert_model_name_or_path'])

tag_complete=json.load(open(os.path.join(save_dir,'label.json'))).split(' ')
tag_encoder=sklearn.preprocessing.LabelEncoder()
tag_encoder.fit(tag_complete)
tag_outside='O'

test_conll_data=get_conll_data(split='test',dir='/home/xhsun/.conll')
test_dataloader=create_dataloader(sentences=test_conll_data.get('sentences'),
                                    tags=test_conll_data.get('tags'),
                                    transformer_tokenizer=transformer_tokenizer,
                                    transformer_config=transformer_config,
                                    max_len=args['max_len'],
                                    tag_encoder=tag_encoder,
                                    tag_outside=tag_outside,
                                    batch_size=args['dev_batch_size'],
                                    num_workers=args['num_workers'])

n_tags=tag_encoder.classes_.shape[0]
model=NERNetwork(model_name_or_path=args['bert_model_name_or_path'],n_tags=n_tags)
success=model.load_state_dict(torch.load(f=os.path.join(save_dir,'pytorch_model.bin'),map_location='cpu'))
print(success)
model.to(device)

predictions=predict(model=model,test_dataloader=test_dataloader,tag_encoder=tag_encoder,device=device)
f1=compute_f1(pred_tags=predictions,golden_tags=test_conll_data.get('tags'),from_test=True)

print("F1 score in test set is {}".format(f1))