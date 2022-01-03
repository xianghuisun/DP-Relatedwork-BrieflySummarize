# DP+NER-Relatedwork-BrieflySummarize
简要总结下依存分析和实体识别的一些数据集以及相关工作

**all_models文件夹下包含所有下列所有提及的模型**

# 数据集

| 数据集                   | 地址                                                         | 来源                                                         | 备注                                                         |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PTB-3.3.0                | [PTB-3.3.0](https://onedrive.live.com/?authkey=%21AP0Ob2Sm%2DO4Y%2DV8&cid=1DCA4A0FD060776E&id=1DCA4A0FD060776E%2160323&parId=1DCA4A0FD060776E%2158828&action=locate) | [链接](https://github.com/wangxinyu0922/Second_Order_Parsing/issues/1) |                                                              |
| ontonotes                | [english](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view)，[Chinese](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view) | [链接](https://github.com/allanj/ner_with_dependency)        | 训练、验证、测试的划分以[Pradhan et al.](https://aclanthology.org/W12-4501.pdf)为标准，因此可以认为是有专用的训练集和测试集的<br>其中训练集：59924、测试集：8262、验证集：8528 |
| GENIA                    | [GENIA](https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data/genia) | [链接]((https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data/genia)) | 没有专用的训练集和测试集                                     |
| CoNLL03                  | [CoNLL03](https://drive.google.com/file/d/1PUH2uw6lkWrWGfl-9wOAG13lvPrvKO25/view) | [链接1](https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md),[链接2](https://github.com/zliucr/CrossNER/tree/main/ner_data/conll2003) | 训练集：14041、测试集：3453、验证集：3250<br>实体类型有四个：PER、LOC、ORG、MISC |
| BioNLP<br>NCBI<br>BC5CDR | [医疗数据集](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) | [链接](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) | 这类数据集大多用于Cross-domain和Multi-task NER               |
| W-NUT                    | [官网](http://noisy-text.github.io/2017/emerging-rare-entities.html) | [链接](https://github.com/leondz/emerging_entities_17)       | 社交媒体类型的NER数据                                        |



# 各个数据集上论文给出的实验结果（取最高得分）

模型分类这一列将所有模型分为3类：

| 模型分类   | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| DP+NER     | 这一类的模型既用了DP标签，也用了NER标签                      |
| NER        | 这一类的模型仅仅用了NER标签                                  |
| Multi-task | 这一类的模型仅仅使用NER标签，额外的任务采用不需要标签的无监督任务 |

## Ontonotes 5.0

| Model                                                        | 模型分类   | P     | R     | F1    |
| ------------------------------------------------------------ | ---------- | ----- | ----- | ----- |
| [A Joint Model for Named Entity Recognition With Sentence-Level Entity Type Attentions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9388880) | Multi-task | 89.85 | 89.22 | 89.53 |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | DP+NER     | 88.59 | 90.17 | 89.88 |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | DP+NER     | 90.14 | 91.58 | 90.85 |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | NER        | 91.1  | 91.5  | 91.3  |
|                                                              |            |       |       |       |

## CoNLL03

| Model                                                        | 模型分类   | P     | R     | F1    |
| ------------------------------------------------------------ | ---------- | ----- | ----- | ----- |
| [Semi-supervised Multitask Learning for Sequence Labeling](https://github.com/marekrei/sequence-labeler) | Multi-task | -     | -     | 86.26 |
| [Empower Sequence Labeling with Task-Aware Neural Language Model](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF) | Multi-task | -     | -     | 91.85 |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | DP+NER     | 92.2  | 92.5  | 92.4  |
| [A Joint Model for Named Entity Recognition With Sentence-Level Entity Type Attentions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9388880) | Multi-task | 92.17 | 92.51 | 92.73 |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | NER        | 93.7  | 93.3  | 93.5  |
| [A Supervised Multi-Head Self-Attention Network for Nested Named Entity Recognition](https://github.com/xyxAda/Attention_NER) | Multi-task | -     | -     | 93.6  |
|                                                              |            |       |       |       |

## GENIA

| Model                                                        | 模型分类   | P     | R    | F1   |
| ------------------------------------------------------------ | ---------- | ----- | ---- | ---- |
| [A Boundary-aware Neural Model for Nested Named Entity Recognition](https://github.com/thecharm/boundary-aware-nested-ner) | Multi-task | 75.9  | 73.6 | 74.7 |
| [A Supervised Multi-Head Self-Attention Network for Nested Named Entity Recognition](https://github.com/xyxAda/Attention_NER) | Multi-task | 80.03 | 78.9 | 79.6 |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | NER        | 81.8  | 79.3 | 80.5 |
|                                                              |            |       |      |      |

**注：因为GENIA是嵌套实体数据集，所以该数据集上的实验模型通常不会考虑依存分析、跨领域。**

**实验结果简要分析：**

1. 每一篇论文尽量都避过了与其他论文相比
2. 各个模型的实验配置大不相同，大多采用ELMo、BERT、char-level embedding以及依存分析依赖的embedding，然后拼接在一起。**（表格中列出的是最优结果，论文中的实验结果表明，如果不用BERT向量，效果下降2~3个点）**
3. 每一个模型所用的具体实验参数配置在后面详细介绍
4. 综合来看，[Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) 这篇论文在所有数据集上效果都是最好的。



# 论文简要总结

## DP+NER

DP+NER这里有三篇论文，其中两篇属于DP+NER，即：既需要DP标签，也需要NER标签。

| 论文                                                         | 会议       | 实验设置                                                   | 备注                                          |
| ------------------------------------------------------------ | ---------- | ---------------------------------------------------------- | --------------------------------------------- |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | EMNLP 2019 | <img width=400/>输入向量为字符向量+依存分析向量+EMLo的拼接 | <img width=400/>用与不用ELMo向量相差1~2个点。 |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | NAACL 2021 | 输入向量为字符向量+依存分析向量+BERT+POS向量的拼接         | 用与不用BERT相差1~2个点                       |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | ACL 2020   | 输入向量为字符向量+fasttext+BERT的拼接                     | 使用了BERT-large，**去掉后下降2.4个点**。     |



## 多任务

多任务这里有3个模型，**均不需要额外的数据来实现多任务**。其中：

- 有两篇是采用语言模型作为辅助任务。
- 有一篇是让模型额外预测句子中有哪些实体标签。

| 论文                                                         | 会议      | 任务1 | 任务2                                    | 实验设置                                                     | 备注                                                         |
| ------------------------------------------------------------ | --------- | ----- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Semi-supervised Multitask Learning for Sequence Labeling](https://github.com/marekrei/sequence-labeler) | ACL 2017  | NER   | LM                                       | <img width=400/>输入向量就是随机初始化的词向量，编码器是BiLSTM | 利用LM任务辅助NER。                                          |
| [Empower Sequence Labeling with Task-Aware Neural Language Model](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF) | AAAI 2018 | NER   | LM                                       | 输入向量Glove，编码器是BiLSTM                                | 也是利用LM辅助NER，与上一篇类似。代码2](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling) |
| [A Joint Model for Named Entity Recognition With Sentence-Level Entity Type Attentions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9388880) | IEEE 2021 | NER   | <img width=400/>预测句子中有哪些实体类型 | 输入向量是词向量+字符向量+BERT拼接                           | 任务2是论文提出的，即：除了预测句子中每一个单词的ner标签，额外预测这个句子中有哪些实体标签，即预测这个句子中是否存在NER、LOC还是ORG等。**这个任务的设计不需要额外数据标注**。 |



## CrossDomain

| 论文                                                         | 会议      | source  | target                                                       | 实验设置                                                     | 备注                                                         |
| ------------------------------------------------------------ | --------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [CrossNER: Evaluating Cross-Domain Named Entity Recognition](https://github.com/zliucr/CrossNER) | AAAI 2021 | CoNLL03 | 论文给出的5个[数据集](https://github.com/zliucr/CrossNER/tree/main/ner_data) | <img width=400/>以政治领域为例：首先在搜集的政治领域语料库上采用掩码语言模型预训练，然后在政治领域的有标注NER数据上微调，来实现Cross-domain。 | <img width=400/>论文收集公布了5个专业领域的NER数据集，而且也提供了对应的领域相关的[预训练语料](https://drive.google.com/drive/folders/1xDAaTwruESNmleuIsln7IaNleNsYlHGn?usp=sharing) |
| [Cross-Domain NER using Cross-Domain Language Modeling](https://github.com/jiachenwestlake/Cross-Domain_NER) | ACL 2019  | CoNLL03 | [医疗数据集](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) |                                                              |                                                              |



# 其它

| 地址                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| https://github.com/yzhangcs/parser                           | 支持很多parsing的功能，可以直接转成Conll形式。**貌似支持将成分分析转换为依存分析的功能。** |
| https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md | NER数据集 （包括ACE04、ACE05、GENIA、CoNLL2003以及Ontonotes），只不过是处理成MRC形式的 |
| https://github.com/juand-r/entity-recognition-datasets       | NER数据集，不局限于英文                                      |
| https://arxiv.org/pdf/1812.09449.pdf                         | A Survey on Deep Learning for Named Entity Recognition       |

