import stanza
import json
import logging
import os
from tqdm import tqdm
from argparse import ArgumentParser
logger=logging.getLogger("main")
logger.setLevel(logging.INFO)

fh=logging.FileHandler('log_Stanza.txt')
fh.setLevel(logging.INFO)

ch=logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
#logger.handlers=[fh]
logger.addHandler(ch)


def main():
    parser = ArgumentParser()
    parser.add_argument("--path_folder",required=True)
    parser.add_argument("--ontonotes5_file",required=True)
    parser.add_argument("--language",default="english")
    args=parser.parse_args()
    lang='en'
    #if args.language=='english':
    #    stanza.download('en')
    #    lang='en'
    #if args.language=="chinese":
    #    stanza.download("zh")
    #    lang='zh'

    nlp = stanza.Pipeline(lang, processors = 'tokenize,mwt,pos,lemma,depparse')
    with open(args.ontonotes5_file) as f:
        data=json.load(f)

    for key in data.keys():
        file_path=os.path.join(args.path_folder,key)
        logger.info("Creating path {}".format(file_path))
        os.makedirs(file_path,exist_ok=True)

    for key,value in data.items():
        parsing_data_of_key=[]
        logger.info("Current is {}".format(key))
        for example in tqdm(value):
            sentence=example['text']
            language=example['language']
            #entities=example['entities']
            if language!=args.language:
                continue
            sentence=sentence.replace('``','').replace("’",'').replace('"','').replace('[','').replace(']','').replace("''",'')
            doc=nlp(sentence)
            for sent_dict in doc.sentences:
                sent_dict=sent_dict.to_dict()
                for each_dict in sent_dict:
                    id_=str(each_dict['id'])
                    form=each_dict['text']
                    lemma='_'
                    postag=each_dict['upos']
                    cpostag=each_dict['upos']
                    feats='_'
                    head=str(each_dict['head'])
                    deprel=each_dict['deprel']
                    
                    parsing_data_of_key.append('\t'.join([id_,form,lemma,postag,cpostag,feats,head,deprel,'_','_']))
                parsing_data_of_key.append('')
        
        logger.info("{} has {} examples".format(key,len(parsing_data_of_key)))
        with open(os.path.join(args.path_folder,key,'ontonotes5_conll'),'w') as f:
            for example in parsing_data_of_key:
                f.write(example+'\n')


if __name__=="__main__":
    main()
    
