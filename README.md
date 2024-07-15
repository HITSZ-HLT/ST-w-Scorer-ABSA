[**中文说明**](https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/) | [**English**](https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/blob/master/README_EN.md)

# ST-w-Scorer-ABSA

本仓库开源了以下论文的代码：

- 标题：Self-Training with Pseudo-Label Scorer for Aspect Sentiment Quad Prediction
- 作者：Yice Zhang, Jie Zeng#, Weiming Hu#, Ziyi Wang#, Shiwei Chen, Ruifeng Xu*
- 会议：ACL-2024 Main (Long)

## 工作简介
### ASQP任务

ASQP (aspect sentiment quad prediction) 任务是 ABSA (aspect-based sentiment analysis) 中最具代表性、最有挑战性的任务，旨在以四元组的形式从评论文本中识别用户方面级别的情感和观点。一个四元组包含以下四个元素：
- aspect term: 方面项，也称评价对象 (opinion target)，是指文本中被评价的实体；
- aspect category: 方面类别，是预定义的类别，反映了被评价的观点目标的具体方面和维度；
- opinion term: 观点项，是表达情感和观点的词或者短语；
- sentiment polarity: 情感倾向，类别空间为`{POS, NEG, NEU}`。

比如给定评论“_the food is great and reasonably priced_”，ASQP任务的输出应为{(_food_, food\_qulaity,_great_,positive), (_food_, food\_prices,_reasonably priced_,positive}。

### 本文的动机和方法

ASQP任务的关键挑战是标记数据的不足，这限制了现有模型的性能。许多研究者使用数据增强方法来缓解这个问题。然而，数据增强方法的一个显著问题是会不可避免地引入文本和标签不一致的样本，这反而会损害模型的学习。

<div align="center"> <img src="https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/assets/9134454/b133ab76-9a63-4ce0-9de4-861f4804bc21" alt="打分器" width="50%" /></div>

为了减少不一致的样本，我们为数据增强方法引入了一个伪标签打分器。如上图所示，该打分器旨在评估文本和伪标签之间的一致性。如果我们有一个足够健壮的打分器，我们就可以将所有不一致的样本过滤掉，因而极大地提高数据增强的有效性。

我们从数据和模型架构两个角度来增强打分器的有效性和可靠性：
- 我们构建了一个人类标注的比较数据集。具体来说，我们首先使用现有的标记数据训练一个ASQP模型，然后使用该模型为无标注数据生成多个伪标签，接下来让标注者从中选择最适合的伪标签作为正标签，并将其他标签作为负标签。此外，我们还探索了使用大语言模型代替人类标注者的可能性。
- 受到最近偏好优化工作的启发，我们将生成模型为伪标签赋予的条件似然作为对其质量的打分。和前人的方法相比，该方法可以逐token地检查伪标签的合理性，因而提供一个更加全面且有效的打分。

### 实验结果

本方法的主要实验结果如下表，详细的分析见论文。

<div align="center"> <img src="https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/assets/9134454/d7df2c87-6701-4897-b9a2-b66e88fcc1ec" alt="Result" width="80%" /></div>

## 运行代码

### 环境配置

- lightning==2.1.3
- numpy==2.0.0
- scikit_learn==1.5.0
- spacy==3.7.4
- torch==2.1.1+cu118
- tqdm==4.66.2
- transformers==4.36.2

### 代码结构

```
├── data
│   ├── comp
│   │   ├── acos
│   │   │   ├── laptop16
│   │   │   └── rest16
│   │   └── asqp
│   │       ├── rest15
│   │       └── rest16
│   ├── raw
│   │   ├── laptop
│   │   └── yelp
│   └── self-training
│   └── t5
│       ├── acos
│       │   ├── laptop16
│       │   └── rest16
│       └── asqp
│           ├── rest15
│           └── rest16
├── bash
│   ├── do_filtering.sh
│   ├── do_reranking.sh
│   ├── pseudo_labeling.sh
│   ├── train_quad_batch_parallel.sh
│   ├── train_quad.sh
│   └── train_scorer.sh
├── read_quad_result.py
├── train_quad.py
├── train_scorer.py
└── utils
    ├── __init__.py
    ├── loss.py
    ├── quad.py
    └── quad_result.py
```
### 运行代码

在`code`目录下
- 运行 `chmod +x bash/*`。
- 训练初始模型 `bash/train_quad.sh -c 0 -d acos/rest16 -b quad -s 42`。
- 伪标注 `bash/pseudo_labeling.sh -c 0 -d acos/rest16 -b quad`。
- 训练打分器 `bash/train_scorer.sh -c 0 -d acos/rest16 -b scorer -s 42 -l 20 -t 01234+ -a 1`。
- 过滤伪标注数据 `bash/do_filtering.sh -c 0 -d acos/rest16 -b scorer。`

这样就可以得到过滤后的伪标注数据了，接下来利用这些数据训练ASQP模型
- 结合伪标注数据训练ASQP模型 `bash/train_quad.sh -c 0 -d acos/rest16 -b 10-40_10000 -f 10-40_10000 -t ../output/filter/acos/rest16.json`。
- 重排序 `bash/do_reranking.sh -c 0 -d acos/rest16 -b scorer -q 10-40_10000 -a 2024-6-21`。

注意
- 如果服务器无法访问huggingface，可以事先将模型权重下载到服务器上，然后将预训练模型的路径设置为模型权重的路径。
- 打分器对最终的性能有重要的影响，调整其训练参数（lr, alpha, batch\_size）是必要的。
- 伪标注之前，需要先解压`/code/data/raw`下的文件。
- 已经过滤好的伪标注数据在`/code/data/self\_training`中。

## 引用我们

    @inproceedings{zhang-etal-2024-self-training-pseudo-label-scorer,  
        title = "Self-Training with Pseudo-Label Scorer for Aspect Sentiment Quad Prediction",  
        author = "Zhang, Yice  and  
          Zeng, Jie  and  
          Hu, Weiming  and  
          Wang, Ziyi  and  
          Chen, Shiwei  and
          Xu, Ruifeng",  
          booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
          month = August,  
          year = "2024",  
          address = "Bangkok, Thailand",  
          publisher = "Association for Computational Linguistics",  
    }  
