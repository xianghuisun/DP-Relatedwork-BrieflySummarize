import stanza
stanza.install_corenlp()
from stanza.server import CoreNLPClient
#text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
import json
import logging
from tqdm import tqdm
import os
from collections import OrderedDict
from argparse import ArgumentParser
logger=logging.getLogger("main")
logger.setLevel(logging.INFO)

fh=logging.FileHandler('log_CoreNLP.txt')
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

    with open(args.ontonotes5_file) as f:
        data=json.load(f)

    for key in data.keys():
        file_path=os.path.join(args.path_folder,key)
        logger.info("Creating path {}".format(file_path))
        os.makedirs(file_path,exist_ok=True)

    with CoreNLPClient(
            annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
            timeout=150000,
            memory='6G',
            be_quiet=True) as client:

        for key,value in data.items():
            f=open(os.path.join(args.path_folder,key,'ontonotes5_conll'),'w')
            parsing_data_of_key=[]
            logger.info("Current is {}".format(key))
            for example in tqdm(value):
                sentence=example['text']
                language=example['language']
                if language!=args.language:
                    continue
                sentence=sentence.replace('``','').replace("’",'').replace('"','').replace('[','').replace(']','').replace("''",'')
                ann=client.annotate(sentence)

                for ann_sentence in ann.sentence:
                    sentence_dict={}
                    all_words_pos=[]
                    for token in ann_sentence.token:
                        all_words_pos.append((token.word,token.pos))
                    assert len(all_words_pos)==len(ann_sentence.basicDependencies.node)
                    for edge in ann_sentence.basicDependencies.edge:
                        source=edge.source
                        target=edge.target
                        dep=edge.dep
                        sentence_dict[target]={'head':source,'dep':dep}
                    if len(ann_sentence.basicDependencies.root)<1:
                        continue
                    sentence_dict[ann_sentence.basicDependencies.root[0]]={"head":0,'dep':"root"}
                    assert len(sentence_dict)==len(all_words_pos)==len(ann_sentence.basicDependencies.node)
                    for id_ in range(1,len(sentence_dict)+1):
                        form=all_words_pos[id_-1][0]
                        postag=all_words_pos[id_-1][1]
                        cpostag=postag
                        lemma='_'
                        feats='_'
                        head=str(sentence_dict[id_]["head"])
                        deprel=sentence_dict[id_]['dep']
                        parsing_data_of_key.append('\t'.join([str(id_),form,lemma,postag,cpostag,feats,head,deprel,'_','_']))
                        f.write('\t'.join([str(id_),form,lemma,postag,cpostag,feats,head,deprel,'_','_'])+'\n')
                    parsing_data_of_key.append('')
                    f.write(''+'\n')

            logger.info("{} has {} examples".format(key,len(parsing_data_of_key)))
            f.close()
            #with open(os.path.join(args.path_folder,key,'ontonotes5_conll'),'w') as f:
            #    for example in parsing_data_of_key:
            #        f.write(example+'\n')
    
    
if __name__=="__main__":
    main()
