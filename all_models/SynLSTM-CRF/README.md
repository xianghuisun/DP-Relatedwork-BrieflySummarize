# Better Feature Integration for Named Entity Recognition

[NAACL 2021] [Better Feature Integration for Named Entity Recognition (In NAACL 2021)](https://arxiv.org/abs/2104.05316)

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

CUDA 10.1, 10.2

[Bert-as-service](https://github.com/hanxiao/bert-as-service)


# Running   

Firstly, download the embedding files: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) , [cc.ca.300.vec, cc.es.300.vec, cc.zh.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html), and put the files in the data folder.

By default, the model eval our saved model (without BERT) on SemEval 2010 Task 1 Spanish dataset.  

```
python main.py  
```

To train the model with other datasets:    
```
python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt
```

To train with BERT, first obtain the contextual embedding with the instructions in the **get_context_emb** folder (The contextual embedding files for OntoNotes Engligh can be downloaded from [***here***](https://drive.google.com/drive/folders/1Eh3RR7QDmrjUhY6MCy7QlAcXPQrRC7Fy).), and then run with the command:
```
python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt --context_emb=bert 
```

Note that the flag **--dep_model=dggcn** (by default) is where we call both GCN and our Syn-LSTM model. The flag **--num_lstm-layer** is designed for running some baselines, and should be set to 0 (by default) when running our proposed model. 

# About Dataset

Note that we use the data from 4 columns: word, dependency head index, dependency relation label, and entity label.


# Related Repo
The code are created based on [the code](https://github.com/allanj/ner_with_dependency) of the paper "Dependency-Guided LSTM-CRF Model for Named Entity Recognition", EMNLP 2019.



| parameter           | value |
| ------------------- | ----- |
| batch_size          | 100   |
| num_epochs          | 60    |
| optimizer           | sgd   |
| embedding_dim       | 100   |
| learning_rate       | 0.2   |
| hidden_dim          | 200   |
| dep_hidden_dim      | 200   |
| dep_emb_size        | 50    |
| num_gcn_layers      | 2     |
| dropout             | 0.5   |
| dep_model           | dggcn |
| char_emb_size       | 30    |
| charlstm_hidden_dim | 50    |



# Result

## Stanza parsing

|                | train        | dev        | test       |
| -------------- | ------------ | ---------- | ---------- |
| Before parsing | 14041        | 3453       | 3250       |
| After parsing  | 11586(-2455) | 2552(-623) | 2830(-698) |

The number in the table represent the number of sentences. -2455 means that there are 14041 sentences in original conll03 train dataset, after Stanza parsing, 2455 sentences can not be used since those sentences are parsed into multiple sentences by Stanza.

#### dev set

| Model    | Prec. | Rec.  | F1    |
| -------- | ----- | ----- | ----- |
| version1 | -     | -     | -     |
| version2 | 95.30 | 95.32 | 95.31 |

#### test set

| Model    | Prec. | Rec.  | F1    |
| -------- | ----- | ----- | ----- |
| version1 | -     | -     | -     |
| version2 | 91.75 | 92.27 | 92.01 |
|          |       |       |       |

- version1 refers to the SynLSTM without BERT
- version2 refers to the DGLSTM with BERT
