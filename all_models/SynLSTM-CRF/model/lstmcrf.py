# 
# @author: Allan
#

import torch
import torch.nn as nn

from config.utils import START, STOP, PAD, log_sum_exp_pytorch
from model.charbilstm import CharBiLSTM
from model.deplabel_gcn import DepLabeledGCN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config.config import DepModelType, ContextEmb, InteractionFunction
import torch.nn.functional as F

device="cuda"
from transformers import AutoTokenizer,AutoModel
tokenizer=AutoTokenizer.from_pretrained("/home/xhsun/NLP/huggingfaceModels/English/bert-base-uncased/")
bert_model=AutoModel.from_pretrained("/home/xhsun/NLP/huggingfaceModels/English/bert-base-uncased/")
bert_model.to(device)
bert_hidden_size=bert_model.config.hidden_size
zero_embedding=torch.zeros([1,bert_hidden_size])

def convert_ids2tokens(idx2word,input_tensor):
    result=[]
    for sentence_ids in input_tensor:
        sentence=[]
        sentence_ids=sentence_ids.cpu().numpy().tolist()
        for each_id in sentence_ids:
            if each_id!=0:
                sentence.append(idx2word[each_id])
        result.append(sentence)
    return result

def solve_wordpiece(last_hidden_state,sen):
    '''
    sen is a sentence in type of string
    last_hidden_state not contains CLS and SEP
    tokenizer.tokenize is not equal to tokenizer.convert_tokens_to_ids
    like sentence : 'Total shares to be offered 0.0 million'
    input_ids :[8653, 6117, 1106, 1129, 2356, 121, 119, 121, 1550]
    word_ids : [8653, 6117, 1106, 1129, 2356, 100, 1550]           100 means UNK
    '''
    assert type(sen)==str
    new_state=[]
    sentence_list=sen.split(' ')
    j=0
    for i in range(len(sentence_list)):
        token=sentence_list[i]
        tokens=tokenizer.tokenize(token)
        piece_length=len(tokens)
        new_state.append(torch.mean(last_hidden_state[j:j+piece_length],dim=0,keepdims=True))
        j+=piece_length
    new_state=torch.vstack(new_state)
    assert new_state.size()==(len(sentence_list),bert_model.config.hidden_size)
    return new_state



class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.dep_model = config.dep_model
        self.context_emb = config.context_emb
        self.interaction_func = config.interaction_func


        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]


        self.input_size = config.embedding_dim#100

        if self.use_char:
            self.char_feature = CharBiLSTM(config)
            self.input_size += config.charlstm_hidden_dim#+50 = 150


        vocab_size = len(config.word2idx)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        if self.dep_model == DepModelType.dglstm and self.interaction_func == InteractionFunction.mlp:
            self.mlp_layers = nn.ModuleList()
            for i in range(config.num_lstm_layer - 1):
                self.mlp_layers.append(nn.Linear(config.hidden_dim, 2 * config.hidden_dim).to(self.device))
            self.mlp_head_linears = nn.ModuleList()
            for i in range(config.num_lstm_layer - 1):
                self.mlp_head_linears.append(nn.Linear(config.hidden_dim, 2 * config.hidden_dim).to(self.device))

        """
            Input size to LSTM description
        """
        self.charlstm_dim = config.charlstm_hidden_dim
        if self.dep_model == DepModelType.dglstm or self.dep_model == DepModelType.dggcn:
            self.input_size += config.dep_emb_size# +50 =200
            # if self.use_char:
            #     self.input_size += config.charlstm_hidden_dim

        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size

        self.input_size += config.dep_emb_size  # +50 =250#
        # graph input is just word emb and char emb
        self.graph_size = config.embedding_dim + config.charlstm_hidden_dim +config.dep_emb_size#100+50+50
        
        self.input_size+=bert_hidden_size#+50 =1018
        print("[Model Info] Input size to LSTM: {}".format(self.input_size))
        print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))#200

        num_layers = 1
        if config.num_lstm_layer > 1 and self.dep_model != DepModelType.dglstm:
            num_layers = config.num_lstm_layer
        if config.num_lstm_layer > 0:
            self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True).to(self.device)

        self.num_lstm_layer = config.num_lstm_layer
        self.lstm_hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim
        if config.num_lstm_layer > 1 and self.dep_model == DepModelType.dglstm:
            self.add_lstms = nn.ModuleList()
            if self.interaction_func == InteractionFunction.concatenation or \
                    self.interaction_func == InteractionFunction.mlp:
                hidden_size = 2 * config.hidden_dim
            elif self.interaction_func == InteractionFunction.addition:
                hidden_size = config.hidden_dim

            print("[Model Info] Building {} more LSTMs, with size: {} x {} (without dep label highway connection)".format(config.num_lstm_layer-1, hidden_size, config.hidden_dim))
            for i in range(config.num_lstm_layer - 1):
                self.add_lstms.append(nn.LSTM(hidden_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device))

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        self.pos_label_embedding = nn.Embedding(len(config.poslabel2idx), config.dep_emb_size).to(self.device)


        final_hidden_dim = config.hidden_dim if self.num_lstm_layer >0 else self.input_size
        """
        Model description
        """
        print("[Model Info] Dep Method: {}, hidden size: {}".format(self.dep_model.name, config.dep_hidden_dim))
        if self.dep_model != DepModelType.none:
            print("Initializing the dependency label embedding")
            self.dep_label_embedding = nn.Embedding(len(config.deplabel2idx), config.dep_emb_size).to(self.device)

            if self.dep_model == DepModelType.dggcn:
                self.gcn = DepLabeledGCN(config, config.hidden_dim, self.input_size, self.graph_size)  ### lstm hidden size
                final_hidden_dim = config.dep_hidden_dim
                #config.hidden_dim==200, input_size==1018, graph_size==200, dep_hidden_dim=50

        print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))

        for key,value in self.__dict__.items():
            print(key,value)
        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)


    def neural_scoring(self, word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, dep_head_tensor, dep_label_tensor, pos_label_tensor, trees=None, config=None):
        """
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param chars: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :param dep_label_tensor: (batch_size, max_sent_len)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        batch_size = word_seq_tensor.size(0)
        sent_len = word_seq_tensor.size(1)

        sentence_lengths=word_seq_lens.cpu().tolist()
        max_length=sentence_lengths[0]
        assert max_length==max([s_l for s_l in sentence_lengths])==sent_len#descending order by sentence length

        bert_embeddings=[]
        if config is not None:
            # print(word_seq_tensor)
            # print(word_seq_lens)
            sentences=convert_ids2tokens(idx2word=config.idx2word,input_tensor=word_seq_tensor)
            #现在的问题是怎么解决wordpiece问题
            for i,sen in enumerate(sentences):
                sen=' '.join(sen)#传递给tokenizer的必须是string或者[string,string]，每一个string represent a sentence
                input_to_bert=tokenizer([sen],return_tensors='pt')#(1,wordpiece_length)
                for key in input_to_bert.keys():
                    input_to_bert[key]=input_to_bert[key].to(self.device)

                with torch.no_grad():
                    last_hidden_state=bert_model(**input_to_bert).last_hidden_state[0]#(wordpiece_length,768)
                    last_hidden_state=last_hidden_state[1:-1,:]#(wordpiece_length-2,768)
                    wordpiece_length=last_hidden_state.size(0)#(wordpiece_length-2)

                    if wordpiece_length!=sentence_lengths[i]:
                        assert wordpiece_length>sentence_lengths[i]
                        last_hidden_state=solve_wordpiece(last_hidden_state,sen)#(sentence_lengths[i],768)

                    assert last_hidden_state.size()==(sentence_lengths[i],bert_hidden_size)
                    
                    if sentence_lengths[i]<max_length:
                        pad_embeddings=zero_embedding.repeat(max_length-sentence_lengths[i],1).to(self.device)
                        last_hidden_state=torch.cat([last_hidden_state,pad_embeddings],dim=0)#(max_length,768)

                    # print(wordpiece_length,max_length,sentence_lengths[i])
                    # print(sen_embed.size(),'-'*100)
                    #print(last_hidden_state.size(),max_length,sentence_lengths[i])
                    assert last_hidden_state.size()==(max_length,bert_hidden_size)
                    bert_embeddings.append(last_hidden_state.unsqueeze(0))

            bert_embeddings=torch.vstack(bert_embeddings)#(bsz,max_length,768)
            assert bert_embeddings.size()==(batch_size,sent_len,bert_model.config.hidden_size)

        word_emb = self.word_embedding(word_seq_tensor)
        if self.use_char:
            # if self.dep_model == DepModelType.dglstm or self.dep_model == DepModelType.dggcn:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lens.cpu())
            word_emb = torch.cat((word_emb, char_features), 2)
        if self.dep_model == DepModelType.dglstm or self.dep_model == DepModelType.dggcn:
            size = self.embedding_dim if not self.use_char else (self.embedding_dim + self.charlstm_dim)
            dep_head_emb = torch.gather(word_emb, 1, dep_head_tensor.view(batch_size, sent_len, 1).expand(batch_size, sent_len, size))

        if self.dep_model == DepModelType.dglstm or self.dep_model == DepModelType.dggcn:
            dep_emb = self.dep_label_embedding(dep_label_tensor)
            word_emb = torch.cat((word_emb, dep_emb), 2)

        pos_emb = self.pos_label_embedding(pos_label_tensor)
        word_emb = torch.cat((word_emb, pos_emb), 2)

        if self.context_emb != ContextEmb.none:
            word_emb = torch.cat((word_emb, batch_context_emb.to(self.device)), 2)

        # if self.use_char:
        #     if self.dep_model != DepModelType.dglstm:
        #         char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lens)
        #         word_emb = torch.cat((word_emb, char_features), 2)

        if bert_embeddings !=[]:
            assert bert_embeddings.size()==(batch_size,sent_len,bert_model.config.hidden_size)
            bert_embeddings=bert_embeddings.to(device=self.device)

            word_emb=torch.cat((word_emb,bert_embeddings),2)

        """
          Word Representation
        """

        word_rep = self.word_drop(word_emb)
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]


        if self.num_lstm_layer > 0:
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, _ = self.lstm(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
            feature_out = self.drop_lstm(lstm_out)
        else:
            feature_out = sorted_seq_tensor

        """
        Higher order interactions
        """
        if self.num_lstm_layer > 1 and (self.dep_model == DepModelType.dglstm):
            for l in range(self.num_lstm_layer-1):
                dep_head_emb = torch.gather(feature_out, 1, dep_head_tensor[permIdx].view(batch_size, sent_len, 1).expand(batch_size, sent_len, self.lstm_hidden_dim))
                if self.interaction_func == InteractionFunction.concatenation:
                    feature_out = torch.cat((feature_out, dep_head_emb), 2)
                elif self.interaction_func == InteractionFunction.addition:
                    feature_out = feature_out + dep_head_emb
                elif self.interaction_func == InteractionFunction.mlp:
                    feature_out = F.relu(self.mlp_layers[l](feature_out) + self.mlp_head_linears[l](dep_head_emb))

                packed_words = pack_padded_sequence(feature_out, sorted_seq_len, True)
                lstm_out, _ = self.add_lstms[l](packed_words, None)
                lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
                feature_out = self.drop_lstm(lstm_out)

        """
        Model forward if we have GCN
        """
        if self.dep_model == DepModelType.dggcn:
            # print("feature input to GCN : ", feature_out.size())
            # print("seq_len : ",sorted_seq_len)
            # print(adj_matrixs)
            feature_out = self.gcn(feature_out, sorted_seq_len, adj_matrixs[permIdx], dep_label_adj[permIdx])

        outputs = self.hidden2tag(feature_out)

        return outputs[recover_idx]

    def calculate_all_scores(self, features):
        batch_size = features.size(0)
        seq_len = features.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    features.view(batch_size, seq_len, 1, self.label_size).expand(batch_size,seq_len,self.label_size, self.label_size)
        return scores

    def forward_unlabeled(self, all_scores, word_seq_lens, masks):
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)

    def forward_labeled(self, all_scores, word_seq_lens, tags, masks):
        '''
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def neg_log_obj(self, words, word_seq_lens, batch_context_emb, chars, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, tags, batch_dep_label, batch_pos_label, trees=None, config=None):
        features = self.neural_scoring(words, word_seq_lens, batch_context_emb, chars, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, batch_dep_label, batch_pos_label, trees, config=config)

        all_scores = self.calculate_all_scores(features)

        batch_size = words.size(0)
        sent_len = words.size(1)

        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)

        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens, mask)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score - labeled_score


    def viterbiDecode(self, all_scores, word_seq_lens):
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        # sent_len =
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx

    def decode(self, batchInput,config=None):
        wordSeqTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, trees, tagSeqTensor, batch_dep_label, batch_pos_label = batchInput
        features = self.neural_scoring(wordSeqTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, batch_dep_label, batch_pos_label, trees, config=config)
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbiDecode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx
