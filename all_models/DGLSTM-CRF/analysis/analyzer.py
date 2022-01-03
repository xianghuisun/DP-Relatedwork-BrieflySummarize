from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List


def read_conll(res_file: str, number: int = -1) -> List[Instance]:
    print("Reading file: " + res_file)
    insts = []
    # vocab = set() ## build the vocabulary
    with open(res_file, 'r', encoding='utf-8') as f:
        words = []
        heads = []
        deps = []
        labels = []
        tags = []
        preds = []
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                inst = Instance(Sentence(words, heads, deps, tags), labels)
                inst.prediction = preds
                insts.append(inst)
                words = []
                heads = []
                deps = []
                labels = []
                tags = []
                preds = []

                if len(insts) == number:
                    break
                continue
            vals = line.split()
            word = vals[1]
            pos = vals[2]
            head = int(vals[3])
            dep_label = vals[4]

            label = vals[5]
            pred_label = vals[6]

            words.append(word)
            heads.append(head)  ## because of 0-indexed.
            deps.append(dep_label)
            tags.append(pos)
            labels.append(label)
            preds.append(pred_label)
    print("number of sentences: {}".format(len(insts)))
    return insts

res_file = "../results/lstm_200_crf_conll2003_-1_dep_none_elmo_1_sgd_gate_0.results"
insts = read_conll(res_file)

total = 0
total_word = 0
for inst in insts:
    gold = inst.output
    prediction = inst.prediction
    words = inst.input.words
    heads = inst.input.heads
    dep_labels = inst.input.dep_labels
    have_error= False
    for idx in range(len(gold)):
        if gold[idx] != 'O' and prediction[idx] == 'O':
            have_error = True
            total_word += 1
            print("{}\t{}\t{}\t{}\t{}\t{}\t".format(idx, words[idx], heads[idx]+1, dep_labels[idx], gold[idx], prediction[idx]))
    if have_error:
        print(words)
        print(gold)
        print(prediction)
        total +=1
        print()
print("number of sentences have errors: {}".format(total))
print("number of words have errors: {}".format(total_word))

