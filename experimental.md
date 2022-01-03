## Ontonotes 5.0

| Model                                                        | 模型分类   | P     | R     | F1    |
| ------------------------------------------------------------ | ---------- | ----- | ----- | ----- |
| [A Joint Model for Named Entity Recognition With Sentence-Level Entity Type Attentions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9388880) | Multi-task | 89.85 | 89.22 | 89.53 |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | DP+NER     | 88.59 | 90.17 | 89.88 |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | DP+NER     | 90.14 | 91.58 | 90.85 |
|                                                              |            |       |       |       |

## CoNLL03

| Model                                                        | 模型分类   | P     | R     | F1    |
| ------------------------------------------------------------ | ---------- | ----- | ----- | ----- |
| [Better Feature Integration for Named Entity Recognition](https://github.com/xuuuluuu/SynLSTM-for-NER) | DP+NER     | 91.75 | 92.27 | 92.01 |
| [Dependency-Guided LSTM-CRF Model for Named Entity Recognition](https://github.com/allanj/ner_with_dependency) | DP+NER     | 91.75 | 92.55 | 92.15 |
| [A Joint Model for Named Entity Recognition With Sentence-Level Entity Type Attentions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9388880) | Multi-task | 92.17 | 92.51 | 92.73 |
|                                                              |            |       |       |       |



**注**

1. 由于CoNLL03没有依存分析的标签，所以采用Stanza进行解析获取依存分析的标签。
2. 还剩下一篇多任务没有实现