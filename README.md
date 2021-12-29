# DP+NER-Relatedwork-BrieflySummarize
简要总结下依存分析和实体识别的一些数据集以及相关工作



# 数据集

| 数据集                   | 地址                                                         | 来源                                                         | 备注                                                         |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PTB-3.3.0                | [PTB-3.3.0](https://onedrive.live.com/?authkey=%21AP0Ob2Sm%2DO4Y%2DV8&cid=1DCA4A0FD060776E&id=1DCA4A0FD060776E%2160323&parId=1DCA4A0FD060776E%2158828&action=locate) | [链接](https://github.com/wangxinyu0922/Second_Order_Parsing/issues/1) |                                                              |
| ontonotes                | [english](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view)，[Chinese](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view) | [链接](https://github.com/allanj/ner_with_dependency)        | 没有专用的训练集和测试集                                     |
| GENIA                    | [GENIA](https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data/genia) | [链接]((https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data/genia)) | 没有专用的训练集和测试集                                     |
| CoNLL03                  | [CoNLL03](https://drive.google.com/file/d/1PUH2uw6lkWrWGfl-9wOAG13lvPrvKO25/view) | [链接1](https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md),[链接2](https://github.com/zliucr/CrossNER/tree/main/ner_data/conll2003) | 训练集：14041、测试集：3453、验证集：3250<br>实体类型有四个：PER、LOC、ORG、MISC |
| BioNLP<br>NCBI<br>BC5CDR | [医疗数据集](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) | [链接](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) | 这类数据集大多用于Cross-domain和Multi-task NER               |





# 论文

## DP+NER

| 论文                                                         | 会议       | 实验数据                                        | 实验结果（P\|R\|F1）                                         | 备注                                              |
| ------------------------------------------------------------ | ---------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | EMNLP 2019 | notonotes<br>CoNLL03                            | 88.53\|88.50\|88.52<br>92.20\|92.50\|92.40                   | <img width=400/>                                  |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | NAACL 2021 | notonotes<br>                                   | 88.96\|89.13\|89.04                                          |                                                   |
| [A Boundary-aware Neural Model for Nested Named Entity Recognition](https://github.com/thecharm/boundary-aware-nested-ner) | ACL 2019   | GENIA                                           | 75.9 \|73.6 \|74.7                                           |                                                   |
| [A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://github.com/foxlf823/sodner) | ACL 2021   | ACE05<br>GENIA                                  | -\|-\|84.3<br>-\|-\|78.3                                     | 论文仅仅给出F1值的结果                            |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | ACL 2020   | ontonotes<br>ACE04<br>ACE05<br>GENIA<br>CoNLL03 | 91.1 \| 91.5 \| 91.3<br>87.3 \| 86.0 \| 86.7<br>85.2 \| 85.6 \| 85.4<br>81.8 \| 79.3 \| 80.5<br>93.7 \| 93.3 \| 93.5 | 使用了BERT-large作为词向量，**去掉后下降2.4个点** |



## CrossDomain

| 论文                                                         | 会议      | source  | target                                                       | 备注                                                         |
| ------------------------------------------------------------ | --------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [CrossNER: Evaluating Cross-Domain Named Entity Recognition](https://github.com/zliucr/CrossNER) | AAAI 2021 | CoNLL03 | 论文给出的5个[数据集](https://github.com/zliucr/CrossNER/tree/main/ner_data) | <img width=400/>论文收集公布了5个专业领域的NER数据集，而且也提供了对应的领域相关的[预训练语料](https://drive.google.com/drive/folders/1xDAaTwruESNmleuIsln7IaNleNsYlHGn?usp=sharing) |
| [Cross-Domain NER using Cross-Domain Language Modeling](https://github.com/jiachenwestlake/Cross-Domain_NER) | ACL 2019  | CoNLL03 | [医疗数据集](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) |                                                              |



## 多任务

| 论文                                                         | 会议      | 任务1 | 任务2               | 备注                                                         |
| ------------------------------------------------------------ | --------- | ----- | ------------------- | ------------------------------------------------------------ |
| Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization | AAAI 2019 | NER   | NEN(命名实体规范化) | 实验所用数据：[NCBI](https://github.com/zhoubaohang/MTAAL/blob/main/dataset/NCBI/train.txt)、[BC5CDR](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-IOB)。这是两个医疗数据集。其中命名实体规范这个任务的标签是作者用工具包生成的，不是人工标注的。 |
| [Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization](https://github.com/zhoubaohang/MTAAL) | AAAI 2021 | NER   | NEN(命名实体规范化) | 实验数据集同上。**这篇论文的效果没有上篇好。**               |
| Joint Learning of Named Entity Recognition and Entity Linking | 不知      | NER   | 实体链接            | **所用数据集是CoNNL03，baseline是92.34，加上多任务后是92.52** |

# 其它Github仓库

| 地址                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| https://github.com/yzhangcs/parser                           | 支持很多parsing的功能，可以直接转成Conll形式。**貌似支持将成分分析转换为依存分析的功能。** |
| https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md | NER数据集 （包括ACE04、ACE05、GENIA、CoNLL2003以及Ontonotes），只不过是处理成MRC形式的 |

