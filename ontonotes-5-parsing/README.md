**该文件夹的作用是将ontonotes5.0中的句子解析出对应的依存分析(dependency parsing)**



# step 1

```bash
python ontonotes5_to_json.py --src path_to_ontonotes-release-5.0_LDC2013T19.tar --dst path_to_save_json --output_file path2_to_save_json
```

- path_to_ontonotes-release-5.0_LDC2013T19.tar指的就是下载的ontonotes-release-5.0_LDC2013T19.tar的路径
- path_to_save_json指的是保存解析出来的json文件路径
- path2_to_save_json指的是另一个解析出来的json文件路径

path_to_save_json与path2_to_save_json不同的是path2_to_save_json的文件是按照不同领域分类的，如：bn、bc、nw、pt、tc、wb



这一步是将ontonotes数据中的sentence和对应的pos、ner等信息解析出来。

# Step2

```bash
python Stanza_parsing.py --path_folder save_file_path_folder --ontonotes5_file path2_to_save_json --language english
```

- path2_to_save_json就是第一步的文件路径
- save_file_path_folder指的是一个文件夹，该文件夹下会创建多个子文件夹，用来分别保存每一个类别(bn、bc、nw、wb、pt等)解析后对应的json文件

如果你不想分类，将所有类的句子都放在一起的话，那么就修改Stanza_parsing.py 中for key,value in data.items()这一行代码，对应的path2_to_save_json换成path_to_save_json





**CoreNLP_parsing.py和Stanza_parsing.py的作用是一样的，只不过一个是用CoreNLP解析，另一个是用Stanza解析出来依存分析**



# 参考

[ontonotes-5-parsing](https://github.com/nsu-ai/ontonotes-5-parsing)

[Stanza](https://github.com/stanfordnlp/stanza)