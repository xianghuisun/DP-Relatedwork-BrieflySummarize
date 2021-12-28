# DP+NER-Relatedwork-BrieflySummarize
简要总结下依存分析(Dependency Parsing)和实体识别的一些数据集以及相关工作



# 数据集

| 数据集    | 地址                                                         | 来源                                                         | 备注 |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| PTB-3.3.0 | https://onedrive.live.com/?authkey=%21AP0Ob2Sm%2DO4Y%2DV8&cid=1DCA4A0FD060776E&id=1DCA4A0FD060776E%2160323&parId=1DCA4A0FD060776E%2158828&action=locate | https://github.com/wangxinyu0922/Second_Order_Parsing/issues/1 |      |
| ontonotes | [english](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view)，[Chinese](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view) | https://github.com/allanj/ner_with_dependency                |      |
| GENIA     | https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data/genia |                                                              |      |
| CoNLL03   | https://drive.google.com/file/d/1PUH2uw6lkWrWGfl-9wOAG13lvPrvKO25/view | https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md |      |





# 论文代码

| 论文                                                         | 备注                                                         | 会议       | 实验数据                                        | 实验结果（P\|R\|F1）                                         | 其它                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | 提供了处理好的ontonotes数据（即利用Standford CoreNLP将ontonotes数据的成分分析转为依存分析） | EMNLP 2019 | notonotes<br>CoNLL03                            | 88.53\|88.50\|88.52<br>92.20\|92.50\|92.40                   |                                                   |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | 该论文使用四个数据集，除了ontonotes的中文和英文外，还用了SemEval2010的Catalan和Spanish。其中ontonotes的中文和英文就是利用上一篇Dependency-Guided LSTM-CRF提供的。此外提供了处理成ConllX形式的Catalan和Spanish两个数据集。 | NAACL 2021 | notonotes<br>                                   | 88.96\|89.13\|89.04                                          |                                                   |
| [A Boundary-aware Neural Model for Nested Named Entity Recognition](https://github.com/thecharm/boundary-aware-nested-ner) | 论文旨在处理嵌套NER，首先是预测实体边界，然后对于每一个可能的边界区域，预测这个区域属于哪个实体类型。代码中给出了GENIA数据。 | ACL 2019   | GENIA                                           | 75.9 \|73.6 \|74.7                                           |                                                   |
| [A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://github.com/foxlf823/sodner) | 穷举所有的可能的span，预测每一个是嵌套还是不连续。同时用GNN提取依存分析的特征。 | ACL 2021   | ACE05<br>GENIA                                  | 论文仅仅给出F1值的结果<br>-\|-\|84.3<br>-\|-\|78.3           |                                                   |
| [Named Entity Recognition as Dependency Parsing](https://github.com/juntaoy/biaffine-ner) | **这篇论文的实验结果在每一个数据集上的效果都非常高，显著高于上述几个论文** | ACL 2020   | ontonotes<br>ACE04<br>ACE05<br>GENIA<br>CoNLL03 | 91.1 \| 91.5 \| 91.3<br>87.3 \| 86.0 \| 86.7<br>85.2 \| 85.6 \| 85.4<br>81.8 \| 79.3 \| 80.5<br>93.7 \| 93.3 \| 93.5 | 使用了BERT-large作为词向量，**去掉后下降2.4个点** |





# 其它Github仓库

| 地址                                                         | 说明                                                         |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| https://github.com/yzhangcs/parser                           | 支持很多parsing的功能，可以直接转成Conll形式。**貌似支持将成分分析转换为依存分析的功能。** |      |
| https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md | NER数据集 （包括ACE04、ACE05、GENIA、CoNLL2003以及Ontonotes），只不过是处理成MRC形式的 |      |
|                                                              |                                                              |      |

