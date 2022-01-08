# baseline

The baseline model uses BERT+Linear. BERT is bert-base-uncased. 

## train

```bash
pip install -r requirements.txt
python -m visdom.server
python main.py --file_path path_to_nerdata --bert_model_name_or_path path_to_bert
```

**Note**

- **path_to_nerdata** refers to the folder where conll data is stored. There should be three files in this folder: train.txt,test.txt,valid.txt
- **bert_model_name_or_path** refers to the folder where bert model is located, like xx/bert-base-uncased
- You need modify the variable **save_dir in main.py**. (save_dir refers to the path where the model and log is saved)



## test

```bash
python test.py
```

**Note**

- You need modify the variable save_dir and the path to conll data in test.py



## result

**The parameter configuration is saved in file args_dict where is under folder save_dir**

### Conll03 dev

| Model       | Precision | Recall | F1    |
| ----------- | --------- | ------ | ----- |
| BERT+Linear | 0.937     | 0.948  | 0.943 |
|             |           |        |       |



### Conll03 test

| Model       | Precision | Recall | F1    |
| ----------- | --------- | ------ | ----- |
| BERT+Linear | 0.892     | 0.912  | 0.902 |
|             |           |        |       |

