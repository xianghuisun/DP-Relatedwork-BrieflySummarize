# DP-Relatedwork-BrieflySummarize
简要总结下依存分析(Dependency Parsing)的一些数据集和相关工作



# Dependency Parsing数据集

| 数据集    | 地址                                                         |      |
| --------- | ------------------------------------------------------------ | ---- |
| PTB-3.3.0 | https://onedrive.live.com/?authkey=%21AP0Ob2Sm%2DO4Y%2DV8&cid=1DCA4A0FD060776E&id=1DCA4A0FD060776E%2160323&parId=1DCA4A0FD060776E%2158828&action=locate |      |
| ontonotes | [english](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view)，[Chinese](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view) |      |
|           |                                                              |      |



# 论文代码

| 论文                                                         | 备注                                                         | 数据下载地址                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | 提供了处理好的ontonotes数据（即利用Standford CoreNLP将ontonotes数据的成分分析转为依存分析） | [english](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view)，[Chinese](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view) |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | 该论文使用四个数据集，除了ontonotes的中文和英文外，还用了SemEval2010的Catalan和Spanish。其中ontonotes的中文和英文就是利用上一篇Dependency-Guided LSTM-CRF提供的。此外提供了处理成ConllX形式的Catalan和Spanish两个数据集 |                                                              |
|                                                              |                                                              |                                                              |

# 其它Github仓库

| 地址                               | 说明                                                         |      |
| ---------------------------------- | ------------------------------------------------------------ | ---- |
| https://github.com/yzhangcs/parser | 支持很多parsing的功能，可以直接转成Conll形式。**貌似支持将成分分析转换为依存分析的功能。** |      |
|         https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md                           |                           NER数据集 （包括ACE04、ACE05、GENIA、CoNLL2003以及Ontonotes）                                  |      |
|                                    |                                                              |      |





# 论文(用Parsing辅助NER)

| 论文                                                         | 会议       | 代码                                          |
| ------------------------------------------------------------ | ---------- | --------------------------------------------- |
| [Efficient Dependency-Guided Named Entity Recognition](https://arxiv.org/pdf/1810.08436.pdf) | AAAI 2017  |                                               |
| [Dependency-Guided LSTM-CRF for Named Entity Recognition](https://arxiv.org/pdf/1909.10148.pdf) | EMNLP 2019 | https://github.com/allanj/ner_with_dependency |
| [Better Feature Integration for Named Entity Recognition](https://aclanthology.org/2021.naacl-main.271.pdf) | NAACL 2021 | https://github.com/xuuuluuu/SynLSTM-for-NER   |
| [A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://arxiv.org/pdf/2106.14373.pdf) | ACL 2021   | https://github.com/foxlf823/sodner            |

# 论文实验

## DGLSTM-CRF
### 模型结构
每一个word用三个向量的拼接表示：$u_i=[w_i;w_h;v_r]$，其中$w_h$是单词i的head word的表示。
而每一个$w_k$又是由这个单词的glove向量加上这个单词的char-level向量构成，char-level向量用Bi-LSTM获取。
$v_r$表示单词h到单词i之间的依赖弧，是一个随机初始化的vector


将$[u_1,u_2,\cdots,u_T]$输入到Bi-LSTM中，得到输出$[h_1,h_2,\cdots,h_T]$，利用交互函数$g(\cdot,\cdot)$捕获单词i和它的head之间的依赖，即：$g(h_i,h_{p_i})$。这里采用MLP作为交互函数。即：$g(h_i,h_{p_i})=ReLU(W_1h_i,W_2h_{p_i})$。最后把交互函数的输出送到第二个Bi-LSTM中，输出向量作为CRF的输入。

### 实验结果
实验数据集是Ontonotes5.0的English和Chinese，以及SemEval-2010 task 1中的Catalan和Spanish。**这四个数据集具有实体和依存分析的标注。**

论文也在CoNLL-2003数据集上进行了实验，由于该数据集是实体识别数据集，因此对应的依存关系采用Spacy工具包进行parsing。**实验结果，DGLSTM-CRF仅仅比BiLSTM-CRF在F1指标上多了0.2个分数，表明如果依存分析的标注质量较差，那么会严重影响该模型的效果**

## Syn-LSTM-CRF
### 模型结构
论文指出DGLSTM存在的问题是，利用依存分析的方式过于简单，即仅仅把一个单词和它的head单词的向量以及弧向量拼接在一起扔进LSTM。
这种方法存在的问题：

1. 提升并不显著
2. 一个单词的依存分析和文本信息是两种特征，DGLSTM这种拼接的方式使得两种特征交互的方式不明显，而且使得两种特征共同决定LSTM的输入门

论文提出一种协作LSTM，Synergied-LSTM，即并不是简单的将依存分析的图结构特征和文本信息特征简单的拼接，而是将两种特征作为两个输入（即LSTM此时的输入包括上一时刻的隐藏状态$h_{t-1}$和当前时刻的输入$x_t$以及$g_t$），为图结构特征单独设置一个门。

每一个单词的输入向量由四个向量拼接，分别是glove词向量、BiLSTM的字符向量、依存分析的弧向量、POS向量。
**首先利用GCN提取图结构特征**，初始时每一个单词的特征向量是由三个向量拼接：分别是glove向量、字符向量、依存分析的弧向量。提取图结构特征的目的是针对每一个单词，汇聚其邻居节点的信息（这里的邻居指的是依存分析中的head）

实验所用数据集和DGLSTM中是一样的，Ontonotes5.0的English和Chinese，以及SemEval-2010 task 1中的Catalan和Spanish。**这四个数据集具有实体和依存分析的标注。**



