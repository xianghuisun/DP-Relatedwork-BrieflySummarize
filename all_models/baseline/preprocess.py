import torch
import warnings
from itertools import compress
import os
import csv
import transformers
import sklearn.preprocessing

class DataSet():
    def __init__(self, 
                sentences: list, 
                tags: list, 
                transformer_tokenizer: transformers.PreTrainedTokenizer, 
                transformer_config: transformers.PretrainedConfig, 
                max_len: int, 
                tag_encoder: sklearn.preprocessing.LabelEncoder, 
                tag_outside: str,
                pad_sequences : bool = True) -> None:
        """Initialize DataSetReader
        Initializes DataSetReader that prepares and preprocesses 
        DataSet for Named-Entity Recognition Task and training.
        Args:
            sentences (list): Sentences.
            tags (list): Named-Entity tags.
            transformer_tokenizer (transformers.PreTrainedTokenizer): 
                tokenizer for transformer.
            transformer_config (transformers.PretrainedConfig): Config
                for transformer model.
            max_len (int): Maximum length of sentences after applying
                transformer tokenizer.
            tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
                for Named-Entity tags.
            tag_outside (str): Special Outside tag. like 'O'
            pad_sequences (bool): Pad sequences to max_len. Defaults
                to True.
        """
        self.sentences = sentences
        self.tags = tags
        self.transformer_tokenizer = transformer_tokenizer
        self.max_len = max_len
        self.tag_encoder = tag_encoder
        self.pad_token_id = transformer_config.pad_token_id
        self.tag_outside_transformed = tag_encoder.transform([tag_outside])[0]
        self.pad_sequences = pad_sequences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        tags = self.tags[item]
        # encode tags
        tags = self.tag_encoder.transform(tags)
        
        # check inputs for consistancy
        assert len(sentence) == len(tags)

        input_ids = []
        target_tags = []
        tokens = []
        offsets = []
        
        # for debugging purposes
        # print(item)
        for i, word in enumerate(sentence):
            # bert tokenization
            wordpieces = self.transformer_tokenizer.tokenize(word)
            tokens.extend(wordpieces)
            # make room for CLS if there is an identified word piece
            if len(wordpieces)>0:
                offsets.extend([1]+[0]*(len(wordpieces)-1))
            # Extends the ner_tag if the word has been split by the wordpiece tokenizer
            target_tags.extend([tags[i]] * len(wordpieces)) 
               
        # Make room for adding special tokens (one for both 'CLS' and 'SEP' special tokens)
        # max_len includes _all_ tokens.
        if len(tokens) > self.max_len-2:
            msg = f'Sentence #{item} length {len(tokens)} exceeds max_len {self.max_len} and has been truncated'
            warnings.warn(msg)
        tokens = tokens[:self.max_len-2] 
        target_tags = target_tags[:self.max_len-2]
        offsets = offsets[:self.max_len-2]

        # encode tokens for BERT
        # TO DO: prettify this.
        input_ids = self.transformer_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.transformer_tokenizer.cls_token_id] + input_ids + [self.transformer_tokenizer.sep_token_id]
        
        # fill out other inputs for model.    
        target_tags = [self.tag_outside_transformed] + target_tags + [self.tag_outside_transformed] 
        attention_mask = [1] * len(input_ids)
        # set to 0, because we are not doing NSP or QA type task (across multiple sentences)
        # token_type_ids distinguishes sentences.
        token_type_ids = [0] * len(input_ids) 
        offsets = [1] + offsets + [1]

        # Padding to max length 
        # compute padding length
        if self.pad_sequences:
            padding_len = self.max_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token_id] * padding_len)
            attention_mask = attention_mask + ([0] * padding_len)  
            offsets = offsets + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_tags = target_tags + ([self.tag_outside_transformed] * padding_len)  
    
        return {'input_ids' : torch.tensor(input_ids, dtype = torch.long),
                'attention_mask' : torch.tensor(attention_mask, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target_tags' : torch.tensor(target_tags, dtype = torch.long),
                'offsets': torch.tensor(offsets, dtype = torch.long)} 
      
def create_dataloader(sentences, 
                      tags, 
                      transformer_tokenizer, 
                      transformer_config, 
                      max_len,  
                      tag_encoder, 
                      tag_outside,
                      batch_size = 1,
                      num_workers = 1,
                      pad_sequences = True):

    if not pad_sequences and batch_size > 1:
        print("setting pad_sequences to True, because batch_size is more than one.")
        pad_sequences = True

    data_reader = DataSet(
        sentences = sentences, 
        tags = tags,
        transformer_tokenizer = transformer_tokenizer, 
        transformer_config = transformer_config,
        max_len = max_len,
        tag_encoder = tag_encoder,
        tag_outside = tag_outside,
        pad_sequences = pad_sequences)
        # Don't pad sequences if batch size == 1. This improves performance.

    data_loader = torch.utils.data.DataLoader(
        data_reader, batch_size = batch_size, num_workers = num_workers
    )

    return data_loader

def get_conll_data(split: str = 'train', 
                   limit: int = None, 
                   dir: str = None) -> dict:
    """Load CoNLL-2003 (English) data split.
    Loads a single data split from the 
    [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) 
    (English) data set.
    Args:
        split (str, optional): Choose which split to load. Choose 
            from 'train', 'valid' and 'test'. Defaults to 'train'.
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned.
        dir (str, optional): Directory where data is cached. If set to 
            None, the function will try to look for files in '.conll' folder in home directory.
    Returns:
        dict: Dictionary with word-tokenized 'sentences' and named 
        entity 'tags' in IOB format.
    Examples:
        Get test split
        >>> get_conll_data('test')
        Get first 5 observations from training split
        >>> get_conll_data('train', limit = 5)
    """
    assert isinstance(split, str)
    splits = ['train', 'valid', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.conll')
    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'
    
    file_path = os.path.join(dir, f'{split}.txt')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'

    # read data from file.
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = '\t')
        for row in reader:
            data.append([row])

    sentences = []
    sentence = []
    entities = []
    tags = []

    for row in data:
        # extract first element of list.
        row = row[0]
        # TO DO: move to data reader.
        if len(row) > 0 and row[0] != '-DOCSTART-':
            sentence.append(row[0])
            tags.append(row[-1])        
        if len(row) == 0 and len(sentence) > 0:
            # clean up sentence/tags.
            # remove white spaces.
            selector = [word != ' ' for word in sentence]#单词!=' '的位置是True
            sentence = list(compress(sentence, selector))#选择sentence中对应位置是True的word
            tags = list(compress(tags, selector))
            # append if sentence length is still greater than zero..
            if len(sentence) > 0:
                sentences.append(sentence)
                entities.append(tags)
            sentence = []
            tags = []
            
   
    if limit is not None:
        sentences = sentences[:limit]
        entities = entities[:limit]
    
    return {'sentences': sentences, 'tags': entities}