---
layout: post
title: "Skip-gram with naiive softmax"
date: 2020-03-03 20:55:40
image: 'https://raw.githubusercontent.com/Thornhill-GYL/markdownpicture/master/word2vec.png'
description: Word2Vec.
category: 'NLP'
tags:
- word2vec
- Skip-gram(naive)
twitter_text: 实现斯坦福cs224deeplearning4nlp的第一章的skip-gram算法实现.
introduction: 实现斯坦福cs224deeplearning4nlp的第一章的skip-gram算法实现。
---



```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
random.seed(1024)
```

```python
print(nltk.__version__)
print(torch.__version__)
```

```
3.4.4
1.3.1
```

### 查看gutenberg中的有那些文件

```python
nltk.corpus.gutenberg.fileids()
```



```
['austen-emma.txt',
 'austen-persuasion.txt',
 'austen-sense.txt',
 'bible-kjv.txt',
 'blake-poems.txt',
 'bryant-stories.txt',
 'burgess-busterbrown.txt',
 'carroll-alice.txt',
 'chesterton-ball.txt',
 'chesterton-brown.txt',
 'chesterton-thursday.txt',
 'edgeworth-parents.txt',
 'melville-moby_dick.txt',
 'milton-paradise.txt',
 'shakespeare-caesar.txt',
 'shakespeare-hamlet.txt',
 'shakespeare-macbeth.txt',
 'whitman-leaves.txt']
```



## 输入数据处理

### Gutenberg用法

1. Gutenberg.sents表示每个句子为最小显示
2. Gutenberg.words表示一个词语为最小显示
3. Gutenberg.raw表示每一个字符为最小显示

#### ps:这部分文件索引位置可能出错，根据提示调整

```python
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100] #作为测试数据
```

```python
corpus = [[word.lower() for word in sent] for sent in corpus] #将corpus中的字母都小写
```

flatten 使得两层数据扁平化

```python
flatten = lambda l: [item for sublist in l for item in sublist]#将两层数据展开为一层，等价于下列函数
# """
# def flattern(l):
#     result = []
#     for sublist in l:
#         for item in sublist:
#             result.append(item)
#     return result
# """
```

### 从unigram分布的尾部提取停用词

```python
word_count = Counter(flatten(corpus))#Counter计算每个字符在列表中出现的次数
word_count.most_common()
```



```
[(',', 96),
 ('.', 66),
 ('the', 58),
 ('of', 36),
 ('and', 35),
 ('--', 27),
 ('"', 26),
 ('."', 26),
 ('to', 25),
 ('-', 24),
 ('a', 21),
 ('in', 20),
 ("'", 20),
 ('s', 19),
 ('his', 17),
 ('that', 17),
 ('whale', 13),
 (';', 12),
 ('is', 12),
 ('by', 10),
 ('he', 10),
 ('with', 10),
 ('or', 10),
 ('for', 10),
 ('it', 9),
 ('which', 9),
 ('great', 9),
 ('this', 8),
 ('as', 8),
 ('leviathan', 8),
 ('sub', 7),
 ('whales', 7),
 ('be', 6),
 ('sea', 6),
 ('him', 5),
 ('all', 5),
 ('take', 5),
 ('...', 5),
 ('up', 5),
 ('but', 5),
 ('one', 5),
 ('!', 5),
 ('ye', 5),
 ('had', 5),
 ('was', 4),
 ('what', 4),
 ('out', 4),
 ('not', 4),
 ('from', 4),
 ('nuee', 4),
 ('extracts', 4),
 ('are', 4),
 ('shall', 4),
 ('your', 4),
 ('there', 4),
 ('on', 4),
 ('like', 4),
 ('(', 3),
 (')', 3),
 ('i', 3),
 ('ever', 3),
 ('old', 3),
 ('world', 3),
 ('you', 3),
 ('called', 3),
 ('maketh', 3),
 ('more', 3),
 ('will', 3),
 ('poor', 3),
 ('have', 3),
 ('long', 3),
 ('whom', 3),
 ('would', 3),
 ('much', 3),
 ('down', 3),
 ('lord', 3),
 ('mouth', 3),
 ('two', 3),
 ('other', 3),
 ('us', 3),
 ('into', 3),
 ('some', 3),
 ('king', 3),
 ('an', 3),
 ('supplied', 2),
 ('usher', 2),
 ('school', 2),
 ('pale', 2),
 ('now', 2),
 ('grammars', 2),
 ('nations', 2),
 ('them', 2),
 ('fish', 2),
 ('our', 2),
 ('through', 2),
 ('true', 2),
 ('dan', 2),
 ('hvalt', 2),
 ('dictionary', 2),
 ('immediately', 2),
 ('latin', 2),
 ('pekee', 2),
 ('devil', 2),
 ('appears', 2),
 ('gone', 2),
 ('earth', 2),
 ('could', 2),
 ('these', 2),
 ('touching', 2),
 ('well', 2),
 ('here', 2),
 ('view', 2),
 ('said', 2),
 ('many', 2),
 ('own', 2),
 ('so', 2),
 ('whose', 2),
 ('thou', 2),
 ('no', 2),
 ('even', 2),
 ('too', 2),
 ('strong', 2),
 ('sit', 2),
 ('grow', 2),
 ('tears', 2),
 ('glasses', 2),
 ('go', 2),
 ('hearts', 2),
 ('who', 2),
 ('before', 2),
 ('strike', 2),
 ('created', 2),
 ('job', 2),
 ('swallow', 2),
 ('jonah', 2),
 ('psalms', 2),
 ('serpent', 2),
 ('thing', 2),
 ('monster', 2),
 ('beast', 2),
 ('gulf', 2),
 ('holland', 2),
 ('most', 2),
 ('among', 2),
 ('we', 2),
 ('days', 2),
 ('monstrous', 2),
 ('history', 2),
 ('country', 2),
 ('very', 2),
 ('their', 2),
 ('were', 2),
 ('let', 2),
 ('fly', 2),
 ('life', 2),
 ('art', 2),
 ('sir', 2),
 ('sperma', 2),
 ('ceti', 2),
 ('[', 1),
 ('moby', 1),
 ('dick', 1),
 ('herman', 1),
 ('melville', 1),
 ('1851', 1),
 (']', 1),
 ('etymology', 1),
 ('late', 1),
 ('consumptive', 1),
 ('grammar', 1),
 ('threadbare', 1),
 ('coat', 1),
 ('heart', 1),
 ('body', 1),
 ('brain', 1),
 ('see', 1),
 ('dusting', 1),
 ('lexicons', 1),
 ('queer', 1),
 ('handkerchief', 1),
 ('mockingly', 1),
 ('embellished', 1),
 ('gay', 1),
 ('flags', 1),
 ('known', 1),
 ('loved', 1),
 ('dust', 1),
 ('somehow', 1),
 ('mildly', 1),
 ('reminded', 1),
 ('mortality', 1),
 ('while', 1),
 ('hand', 1),
 ('others', 1),
 ('teach', 1),
 ('name', 1),
 ('tongue', 1),
 ('leaving', 1),
 ('ignorance', 1),
 ('letter', 1),
 ('h', 1),
 ('almost', 1),
 ('alone', 1),
 ('signification', 1),
 ('word', 1),
 ('deliver', 1),
 ('hackluyt', 1),
 ('sw', 1),
 ('hval', 1),
 ('animal', 1),
 ('named', 1),
 ('roundness', 1),
 ('rolling', 1),
 ('arched', 1),
 ('vaulted', 1),
 ('webster', 1),
 ('dut', 1),
 ('ger', 1),
 ('wallen', 1),
 ('walw', 1),
 ('ian', 1),
 ('roll', 1),
 ('wallow', 1),
 ('richardson', 1),
 ('ketos', 1),
 ('greek', 1),
 ('cetus', 1),
 ('whoel', 1),
 ('anglo', 1),
 ('saxon', 1),
 ('danish', 1),
 ('wal', 1),
 ('dutch', 1),
 ('hwal', 1),
 ('swedish', 1),
 ('icelandic', 1),
 ('english', 1),
 ('baleine', 1),
 ('french', 1),
 ('ballena', 1),
 ('spanish', 1),
 ('fegee', 1),
 ('erromangoan', 1),
 ('librarian', 1),
 (').', 1),
 ('seen', 1),
 ('mere', 1),
 ('painstaking', 1),
 ('burrower', 1),
 ('grub', 1),
 ('worm', 1),
 ('vaticans', 1),
 ('street', 1),
 ('stalls', 1),
 ('picking', 1),
 ('whatever', 1),
 ('random', 1),
 ('allusions', 1),
 ('anyways', 1),
 ('find', 1),
 ('any', 1),
 ('book', 1),
 ('whatsoever', 1),
 ('sacred', 1),
 ('profane', 1),
 ('therefore', 1),
 ('must', 1),
 ('every', 1),
 ('case', 1),
 ('at', 1),
 ('least', 1),
 ('higgledy', 1),
 ('piggledy', 1),
 ('statements', 1),
 ('however', 1),
 ('authentic', 1),
 ('veritable', 1),
 ('gospel', 1),
 ('cetology', 1),
 ('far', 1),
 ('ancient', 1),
 ('authors', 1),
 ('generally', 1),
 ('poets', 1),
 ('appearing', 1),
 ('solely', 1),
 ('valuable', 1),
 ('entertaining', 1),
 ('affording', 1),
 ('glancing', 1),
 ('bird', 1),
 ('eye', 1),
 ('has', 1),
 ('been', 1),
 ('promiscuously', 1),
 ('thought', 1),
 ('fancied', 1),
 ('sung', 1),
 ('generations', 1),
 ('including', 1),
 ('fare', 1),
 ('thee', 1),
 ('commentator', 1),
 ('am', 1),
 ('belongest', 1),
 ('hopeless', 1),
 ('sallow', 1),
 ('tribe', 1),
 ('wine', 1),
 ('warm', 1),
 ('sherry', 1),
 ('rosy', 1),
 ('sometimes', 1),
 ('loves', 1),
 ('feel', 1),
 ('devilish', 1),
 ('convivial', 1),
 ('upon', 1),
 ('say', 1),
 ('bluntly', 1),
 ('full', 1),
 ('eyes', 1),
 ('empty', 1),
 ('altogether', 1),
 ('unpleasant', 1),
 ('sadness', 1),
 ('give', 1),
 ('subs', 1),
 ('how', 1),
 ('pains', 1),
 ('please', 1),
 ('thankless', 1),
 ('clear', 1),
 ('hampton', 1),
 ('court', 1),
 ('tuileries', 1),
 ('gulp', 1),
 ('hie', 1),
 ('aloft', 1),
 ('royal', 1),
 ('mast', 1),
 ('friends', 1),
 ('clearing', 1),
 ('seven', 1),
 ('storied', 1),
 ('heavens', 1),
 ('making', 1),
 ('refugees', 1),
 ('pampered', 1),
 ('gabriel', 1),
 ('michael', 1),
 ('raphael', 1),
 ('against', 1),
 ('coming', 1),
 ('splintered', 1),
 ('together', 1),
 ('unsplinterable', 1),
 ('god', 1),
 ('genesis', 1),
 ('path', 1),
 ('shine', 1),
 ('after', 1),
 ('think', 1),
 ('deep', 1),
 ('hoary', 1),
 ('prepared', 1),
 ('ships', 1),
 ('hast', 1),
 ('made', 1),
 ('play', 1),
 ('therein', 1),
 ('day', 1),
 ('sore', 1),
 ('sword', 1),
 ('punish', 1),
 ('piercing', 1),
 ('crooked', 1),
 ('slay', 1),
 ('dragon', 1),
 ('isaiah', 1),
 ('soever', 1),
 ('besides', 1),
 ('cometh', 1),
 ('within', 1),
 ('chaos', 1),
 ('boat', 1),
 ('stone', 1),
 ('goes', 1),
 ('incontinently', 1),
 ('foul', 1),
 ('perisheth', 1),
 ('bottomless', 1),
 ('paunch', 1),
 ('plutarch', 1),
 ('morals', 1),
 ('indian', 1),
 ('breedeth', 1),
 ('biggest', 1),
 ('fishes', 1),
 (':', 1),
 ('whirlpooles', 1),
 ('balaene', 1),
 ('length', 1),
 ('four', 1),
 ('acres', 1),
 ('arpens', 1),
 ('land', 1),
 ('pliny', 1),
 ('scarcely', 1),
 ('proceeded', 1),
 ('when', 1),
 ('about', 1),
 ('sunrise', 1),
 ('monsters', 1),
 ('appeared', 1),
 ('former', 1),
 ('size', 1),
 ('came', 1),
 ('towards', 1),
 ('open', 1),
 ('mouthed', 1),
 ('raising', 1),
 ('waves', 1),
 ('sides', 1),
 ('beating', 1),
 ('foam', 1),
 ('tooke', 1),
 ('lucian', 1),
 ('visited', 1),
 ('also', 1),
 ('catching', 1),
 ('horse', 1),
 ('bones', 1),
 ('value', 1),
 ('teeth', 1),
 ('brought', 1),
 ('best', 1),
 ('catched', 1),
 ('forty', 1),
 ('eight', 1),
 ('fifty', 1),
 ('yards', 1),
 ('six', 1),
 ('killed', 1),
 ('sixty', 1),
 ('octher', 1),
 ('verbal', 1),
 ('narrative', 1),
 ('taken', 1),
 ('alfred', 1),
 ('d', 1),
 ('890', 1),
 ('whereas', 1),
 ('things', 1),
 ('whether', 1),
 ('vessel', 1),
 ('enter', 1),
 ('dreadful', 1),
 ('lost', 1),
 ('swallowed', 1),
 ('gudgeon', 1),
 ('retires', 1),
 ('security', 1),
 ('sleeps', 1),
 ('montaigne', 1),
 ('apology', 1),
 ('raimond', 1),
 ('sebond', 1),
 ('nick', 1),
 ('me', 1),
 ('if', 1),
 ('described', 1),
 ('noble', 1),
 ('prophet', 1),
 ('moses', 1),
 ('patient', 1),
 ('rabelais', 1),
 ('liver', 1),
 ('cartloads', 1),
 ('stowe', 1),
 ('annals', 1),
 ('seas', 1),
 ('seethe', 1),
 ('boiling', 1),
 ('pan', 1),
 ('bacon', 1),
 ('version', 1),
 ('bulk', 1),
 ('ork', 1),
 ('received', 1),
 ('nothing', 1),
 ('certain', 1),
 ('they', 1),
 ('exceeding', 1),
 ('fat', 1),
 ('insomuch', 1),
 ('incredible', 1),
 ('quantity', 1),
 ('oil', 1),
 ('extracted', 1),
 ('ibid', 1),
 ('death', 1),
 ('sovereignest', 1),
 ('parmacetti', 1),
 ('inward', 1),
 ('bruise', 1),
 ('henry', 1),
 ('hamlet', 1),
 ('secure', 1),
 ('skill', 1),
 ('leach', 1),
 ('mote', 1),
 ('availle', 1),
 ('returne', 1),
 ('againe', 1),
 ('wound', 1),
 ('worker', 1),
 ('lowly', 1),
 ('dart', 1),
 ('dinting', 1),
 ('breast', 1),
 ('bred', 1),
 ('restless', 1),
 ('paine', 1),
 ('wounded', 1),
 ('shore', 1),
 ('flies', 1),
 ('thro', 1),
 ('maine', 1),
 ('faerie', 1),
 ('queen', 1),
 ('immense', 1),
 ('motion', 1),
 ('vast', 1),
 ('bodies', 1),
 ('can', 1),
 ('peaceful', 1),
 ('calm', 1),
 ('trouble', 1),
 ('ocean', 1),
 ('til', 1),
 ('boil', 1),
 ('william', 1),
 ('davenant', 1),
 ('preface', 1),
 ('gondibert', 1),
 ('spermacetti', 1),
 ('men', 1),
 ('might', 1),
 ('justly', 1),
 ('doubt', 1),
 ('since', 1),
 ('learned', 1),
 ('hosmannus', 1),
 ('work', 1),
 ('thirty', 1),
 ('years', 1),
 ('saith', 1),
 ('plainly', 1),
 ('nescio', 1),
 ('quid', 1),
 ('t', 1),
 ('browne', 1),
 ('vide', 1),
 ('v', 1),
 ('e', 1),
 ('spencer', 1),
 ('talus', 1),
 ('modern', 1),
 ('flail', 1),
 ('threatens', 1),
 ('ruin', 1),
 ('ponderous', 1),
 ('tail', 1),
 ('fixed', 1),
 ('jav', 1),
 ('lins', 1),
 ('side', 1),
 ('wears', 1),
 ('back', 1),
 ('grove', 1),
 ('pikes', 1),
 ('waller', 1),
 ('battle', 1),
 ('summer', 1),
 ('islands', 1),
 ('commonwealth', 1),
 ('state', 1),
 ('--(', 1),
 ('civitas', 1),
 ('artificial', 1),
 ('man', 1)]
```



```python
border = int(len(word_count)*0.01)
```

```python
stopwords = word_count.most_common()[:border]+list(reversed(word_count.most_common()))[:border]#选取最多的和最少的前五个
print(stopwords)
```

```
[(',', 96), ('.', 66), ('the', 58), ('of', 36), ('and', 35), ('man', 1), ('artificial', 1), ('civitas', 1), ('--(', 1), ('state', 1)]
```



```python
stopwords = [s[0] for s in stopwords]
stopwords
```



```
[',', '.', 'the', 'of', 'and', 'man', 'artificial', 'civitas', '--(', 'state']
```



#### 创建词典，确定每个word的位置,实际上是一个one-hot 的过程

```python
vocab = list(set(flatten(corpus)) - set(stopwords))#初始字典为集合corpus减去stopwords
vocab.append('<UNK>')#用于生成字典
```

```python
print(len(set(flatten(corpus))),len(vocab))
print(vocab)
```

```
592 583
['you', 'monsters', 'gulf', 'swedish', 'oil', 'wallen', 'bluntly', 'life', 'when', 'created', 'feel', 'bodies', 'picking', 'beast', 'grammar', 'mortality', 'version', 'signification', 'eyes', 'inward', 'nations', 'animal', 'dinting', 'even', 'gospel', 'solely', 'best', 'somehow', 'six', 'clearing', 'two', 'that', 'foam', 'fish', 'psalms', 'killed', 'verbal', 'exceeding', 'subs', 'waves', 'sword', 'doubt', 'since', 'whether', 'work', 'play', 'back', 'mildly', 'perisheth', 'through', 'leviathan', 'proceeded', 'maketh', 'vessel', 'most', 'were', 'after', 'whereas', 'd', 'flail', 'had', 'think', 'dust', 'slay', 'bulk', 'gudgeon', 'what', 'have', 'loves', 'see', 'storied', 'wounded', 'montaigne', 'commentator', 'crooked', 'is', 'one', 'known', 'appearing', 'job', ':', 'give', 'came', 'also', 'talus', 'moses', 'seen', 'narrative', 'arched', 'dreadful', 'til', 'hie', 'dart', 'forty', 'these', 'paine', 'promiscuously', 'saxon', 'there', 'pan', 'poets', 'received', 'whales', 'hopeless', 'rolling', 'nothing', 'cetus', 'on', 'bacon', 'full', 'pekee', 'wallow', 'anyways', 'dusting', 'ian', 'letter', 'thro', 'isaiah', 'grammars', 'sometimes', 'plutarch', 'tribe', 'monstrous', '."', '(', 'liver', 'appears', 'v', 'handkerchief', 'said', 'fishes', 'raimond', 'this', 'boat', 'punish', 'greek', 'whatever', 'prophet', 'tuileries', 'stalls', 'path', 'eye', 'am', 'size', 'by', 'besides', 'mouthed', 'e', 'faerie', 'day', 'former', 'hval', 'fare', 'fat', ']', 'parmacetti', 'soever', 'monster', 'yards', 'foul', 'me', 'immediately', 'authors', 'sub', 'seas', 'loved', 'fegee', 'dictionary', 'aloft', 'gabriel', 'say', 'name', 'ignorance', 'walw', 'pikes', 'calm', 'queer', 'webster', 'shall', 'nuee', 'word', 'it', 'no', 'has', 'fifty', 'sides', 'grove', 'thought', 'indian', 'cetology', 'can', 'generally', 'erromangoan', 'against', 'jav', 'ever', 'wine', 'reminded', 'heart', 'old', 'must', 'royal', 'an', 'catching', '890', 'late', 'insomuch', 'fly', 'called', 'ibid', 'gondibert', 'teach', 'pliny', 'pale', 'including', 'modern', 'chaos', 'hackluyt', 'would', 'however', 'embellished', 'named', 'true', 'from', 'quid', 'all', 'serpent', 'worm', 'at', 'who', 'or', 'a', 'almost', 'allusions', 'roll', 'book', 'sperma', 'him', 'flies', 'we', 'scarcely', 'but', 'court', 'melville', 'rosy', '[', 'latin', 'not', 'coat', 'glasses', 'shore', 'horse', 'sovereignest', 'much', 'warm', 'please', 'incredible', 'open', 'out', 'things', 'skill', 'roundness', '"', 'whom', 'pains', 'annals', 'certain', 'cometh', 'land', 'vaticans', 'noble', 'thirty', 'restless', 'your', 'danish', 'saith', 'threadbare', 'sleeps', 'alfred', 'alone', 'tears', 'mockingly', 'whale', 'extracts', 'upon', 'returne', 'be', 'earth', 'statements', 'hosmannus', 'againe', 'swallow', 'plainly', 'his', 'king', 'therein', 'richardson', 'veritable', 'described', 'too', 'hamlet', 'belongest', 'bottomless', 'boil', 'value', 'dan', 'waller', 'rabelais', 'peaceful', 'could', 'brought', ';', 'english', 'for', 'lexicons', 'holland', 'sallow', 'they', 'generations', 'thee', 'which', 'spanish', ')', 'battle', 'fancied', 'friends', 'herman', 'hast', 'was', 'some', 'mast', 'hoary', 'strong', 'queen', 'ponderous', 'years', 'bruise', 'empty', 'ceti', '1851', 'down', 'long', 'grow', 'sore', 'summer', 'swallowed', 'thing', 'visited', 'sacred', 'side', 'sw', 'into', 'thankless', 'lost', 'bones', 'men', 'splintered', 'availle', 'genesis', 'up', 'mote', 'ork', 'vast', 'wal', 'any', 'affording', 'touching', '--', 'biggest', 'deliver', 'he', 'now', 'piggledy', 'threatens', 'balaene', 'to', 'consumptive', 'sherry', 'heavens', 'commonwealth', 'mere', 'raphael', 'patient', 'motion', 'higgledy', 'many', 'grub', 'sir', 'before', 'usher', 'baleine', 'lowly', 'painstaking', 'random', 'about', 'profane', 'gulp', 'hvalt', 'street', 'world', 'very', 'stowe', 'henry', 'acres', 'secure', 'wound', 'strike', 'view', 'maine', 'will', 'more', 'vide', "'", 'whoel', 'deep', 'unpleasant', 'within', 'glancing', 'case', 'appeared', 'least', 'security', 'piercing', 'sit', 'tooke', 'us', 'ocean', 'god', 'sea', 'prepared', 'four', 'moby', 'ruin', 'tongue', 'lins', 'bred', ').', 'valuable', 'anglo', 'wears', 'sung', 'spermacetti', 'ketos', 'french', 'hearts', 'towards', 'learned', 'preface', 'lord', 'dut', 'nick', 'nescio', 'convivial', 'teeth', 'whirlpooles', 'flags', 'morals', 'burrower', 'refugees', 'mouth', 'dragon', 'breedeth', 'clear', 'dutch', 'brain', '!', 'justly', 'among', 'gay', 'sixty', 'boiling', 'far', 'etymology', 'with', 'entertaining', 'thou', 'eight', 'let', 'bird', 'retires', 'breast', 'trouble', 'great', 'spencer', 'others', 'ancient', 't', 'made', 'enter', 'while', 'making', 'vaulted', '-', 'here', 'raising', 'browne', 'hampton', 'immense', 'their', 'librarian', 'length', 'lucian', 'beating', 's', 'gone', 'stone', 'ships', 'find', 'own', 'hwal', 'how', 'hand', 'michael', 'so', 'octher', 'are', 'fixed', 'davenant', 'seethe', 'whose', 'country', 'poor', 'tail', 'ger', 'worker', 'pampered', 'therefore', 'if', 'take', 'leaving', '...', 'other', 'islands', 'devil', 'ye', 'i', 'might', 'catched', 'goes', 'paunch', 'school', 'body', 'our', 'quantity', 'in', 'sebond', 'like', 'whatsoever', 'seven', 'apology', 'h', 'authentic', 'every', 'shine', 'jonah', 'arpens', 'icelandic', 'been', 'incontinently', 'cartloads', 'supplied', 'sunrise', 'days', 'well', 'devilish', 'as', 'ballena', 'them', 'death', 'leach', 'sadness', 'altogether', 'history', 'go', 'extracted', 'art', 'dick', 'taken', 'together', 'william', 'coming', 'unsplinterable', '<UNK>']
```



```python
#这是具体代码实现one-hot encoding的过程
word2index = {'<UNK>':0}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo]=len(word2index)
```

```python
index2word = {v:k for k, v in word2index.items()} 
```

## 准备训练数据

```python
WINDOW_SIZE = 3 #代表了输入单词前面有三个单词，后面有三个单词
```

```python
# under_window = []
# for c in corpus:
#     print(c)
#     under_window = list(nltk.ngrams(['<DUMMY>']* WINDOW_SIZE + c +['<DUMMY>']*WINDOW_SIZE,WINDOW_SIZE*2+1))
#     print(under_window)
```

```python
windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])
```

```python
windows[0]#生成每一个处理窗口，方便后面生成二元组
```



```
('<DUMMY>', '<DUMMY>', '<DUMMY>', '[', 'moby', 'dick', 'by')
```



```python
train_data = []
for window in windows:
    for i in range(WINDOW_SIZE*2+1):
        if i ==WINDOW_SIZE or window[i]=='<DUMMY>':#第一部分是要将中心词作为第一个；第二部分是去除DUMMY
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))#形成以中心词，和在窗口附近的词的二元组
print(train_data[:WINDOW_SIZE*2])                      
```

```
[('[', 'moby'), ('[', 'dick'), ('[', 'by'), ('moby', '['), ('moby', 'dick'), ('moby', 'by')]
```

### 该函数可获得对应单词位置的tensor

```python
def prepare_word(word, word2index):
    return Variable(torch.LongTensor([word2index[word]]) if word2index.get(word) is not None else torch.LongTensor([word2index["<UNK>"]]))
```

```python
X_p = []
Y_p = []
train_data[0]
```



```
('[', 'moby')
```



### X_p是中心词对应的tensor,Y_p是相应的target word 对应的tensor

```python
for tr in train_data:
    X_p.append(prepare_word(tr[0],word2index).view(1,-1))#(1,-1)确定1行，n列
    Y_p.append(prepare_word(tr[1],word2index).view(1,-1))
```

```python
train_data = list(zip(X_p,Y_p))#将数据合并为字典，等一下方便训练
```

```python
len(train_data)
```



```
7606
```



## 构建Skip-gram模型

```python
class Skipgram(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(Skipgram,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)
        
        self.embedding_v.weight.data.uniform_(-1,1)#均匀分布来初始化权重均值为-1，方差为1
        self.embedding_u.weight.data.uniform_(0,0)#均匀分布来初始话权重均值为0，方差为0
    
    def forward(self,center_words,target_words,outer_words):
        #这个地方需要注意的是embedding后是行向量，而公式中给出的是列向量，
        #所以center需要转置，另外两个变量在公式中为原向量的转置，所以现在则不用进行处理
        center_embeds = self.embedding_v(center_words)#将中心词进行embedding B x 1 x D
        target_embeds = self.embedding_u(target_words)#将目标单词进行embedding B x 1 x D
        outer_embeds = self.embedding_u(outer_words)#其他的在中心词出现的情况下，出现的单词的embeddingB x V x D
        
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxVxD * BxDx1 => BxV
        #下面是其loss函数
        nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
        #torch.sum(input,1)按行求和，求和之后需要增加一个维度，进行相除
        
        return nll# negative log likelihood
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds 
```

## 开始训练

```python
EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 100
```

```python
losses = []
model = Skipgram(len(word2index), EMBEDDING_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

```python
print(model)
```

```
Skipgram(
  (embedding_v): Embedding(583, 30)
  (embedding_u): Embedding(583, 30)
)
```

### 获取batch训练数据的函数

```python
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
```

### 此函数获取词典中各个单词的位置，用于构建outer_words,等价理解函数在下方

```python
def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(torch.LongTensor(idxs))
```

```python
# idx=[]
# for w in list(vocab):
#     if word2index[w] is not None:
#         idx.append(word2index.get(w))
# Variable(torch.LongTensor(idx))
```

```python
for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        
        inputs, targets = zip(*batch)#解压batch ,将X_p赋值给inputs,将Y_p赋值给targets
        inputs = torch.cat(inputs) # B x 1
        targets = torch.cat(targets)#B x 1
        vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab))  # 将获得的vocabs变为B x V的向量
        
        model.zero_grad()
        
        loss = model(inputs,targets,vocabs)#这里用的是自定义的loss 函数
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())#在0.4版本中有0维的标量，直接用loss.item()得到其 loss 的数值就可以了。

    if epoch % 10 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch,np.mean(losses)))
        losses = []
```

```
Epoch : 0, mean_loss : 5.02
Epoch : 10, mean_loss : 4.02
Epoch : 20, mean_loss : 3.42
Epoch : 30, mean_loss : 3.30
Epoch : 40, mean_loss : 3.25
Epoch : 50, mean_loss : 3.23
Epoch : 60, mean_loss : 3.22
Epoch : 70, mean_loss : 3.21
Epoch : 80, mean_loss : 3.21
Epoch : 90, mean_loss : 3.20
```

## 开始实现test 部分

```python
def word_similarity(target, vocab):
    #现在target就是中心词
    target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target: continue
        
        vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0] #使用余弦计算单词间的距离来表示其他单词
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10] # sort by similarity最相似的10的单词

```

```python
test = random.choice(list(vocab))
test
```



```
'mast'
```



```python
word_similarity(test, vocab)
```



```
[['raphael', 0.6221854090690613],
 ['embellished', 0.5924732685089111],
 ['gay', 0.5921356678009033],
 ['against', 0.5544277429580688],
 ['your', 0.5519949197769165],
 ['with', 0.5506114959716797],
 ['mockingly', 0.5466769337654114],
 ['coming', 0.5152226090431213],
 ['aloft', 0.5016989707946777],
 ['tears', 0.4948738217353821]]
```










