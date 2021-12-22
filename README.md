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
|                                    |                                                              |      |
|                                    |                                                              |      |



# NER数据集

https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md



# 论文(用Parsing辅助NER)

| 论文                                                         | 会议       | 代码                                          |
| ------------------------------------------------------------ | ---------- | --------------------------------------------- |
| [Efficient Dependency-Guided Named Entity Recognition](https://arxiv.org/pdf/1810.08436.pdf) | AAAI 2017  |                                               |
| [Dependency-Guided LSTM-CRF for Named Entity Recognition](https://arxiv.org/pdf/1909.10148.pdf) | EMNLP 2019 | https://github.com/allanj/ner_with_dependency |
| [Better Feature Integration for Named Entity Recognition](https://aclanthology.org/2021.naacl-main.271.pdf) | NAACL 2021 | https://github.com/xuuuluuu/SynLSTM-for-NER   |
| [A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://arxiv.org/pdf/2106.14373.pdf) | ACL 2021   | https://github.com/foxlf823/sodner            |

