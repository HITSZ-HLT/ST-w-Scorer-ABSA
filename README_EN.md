[**中文说明**](https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/) | [**English**](https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/blob/master/README_EN.md)

# ST-w-Scorer-ABSA

This repository releases the code of the following paper:
- Title: Self-Training with Pseudo-Label Scorer for Aspect Sentiment Quad Prediction
- Authors: Yice Zhang, Jie Zeng#, Weiming Hu#, Ziyi Wang#, Shiwei Chen, Ruifeng Xu*
- Conference: ACL-2024 Main (Long)

## Brief Introduction of Our Paper

### The ASQP Task

The ASQP (Aspect Sentiment Quad Prediction) task is the most representative and challenging task in ABSA (Aspect-Based Sentiment Analysis), aiming to identify user-level sentiments and opinions from review texts in the form of quads. Each quad contains the following four elements:
- Aspect term: Also known as the opinion target, this refers to the entity being evaluated in the text.
- Aspect category: This is a predefined category that reflects the specific aspects and dimensions of the evaluated opinion target.
- Opinion term: This is the word or phrase that expresses sentiments and opinions.
- Sentiment polarity: The category space for sentiment polarity is `{POS, NEG, NEU}` (positive, negative, neutral).

For example, given the review "_the food is great and reasonably priced_", the output of the ASQP task should be {(_food_, food_quality, _great_, positive), (_food_, food_prices, _reasonably priced_, positive)}.

### Motivation and Methodology

The key challenge in the ASQP (Aspect Sentiment Quad Prediction) task is the scarcity of labeled data, which limits the performance of existing models. Many researchers have used data augmentation techniques to mitigate this issue. However, a significant problem with data augmentation methods is that they inevitably introduce samples where the text and labels are inconsistent, which can hinder the learning process of the model.

To reduce the occurrence of inconsistent samples, we introduced a pseudo-label scorer for the data augmentation methods. As shown in the image above, this scorer is designed to assess the consistency between the text and pseudo-labels. If we have a robust scorer, we can filter out all inconsistent samples, thereby greatly enhancing the effectiveness of data augmentation.

We enhanced the effectiveness and reliability of the scorer from two perspectives: data and model architecture:
- We built a human-annotated comparison dataset. Specifically, we first trained an ASQP model using existing labeled data, then used this model to generate multiple pseudo-labels for unlabeled data. Subsequently, annotators selected the most suitable pseudo-label as the positive label, and treated the other labels as negative. Additionally, we explored the possibility of using large language models as a substitute for human annotators.
- Inspired by recent work on preference optimization, we used the conditional likelihood assigned by the generative model to pseudo-labels as a score for their quality. Compared to previous methods, this approach allows for token-by-token examination of the pseudo-labels' validity, providing a more comprehensive and effective scoring mechanism.

### Experimental Results

The main experimental results of this method are summarized in the table below. For a detailed analysis, please refer to the paper.

<div align="center"> <img src="https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/assets/9134454/d7df2c87-6701-4897-b9a2-b66e88fcc1ec" alt="Result" width="80%" /></div>

## How to Run

### Requirements

- lightning==2.1.3
- numpy==2.0.0
- scikit_learn==1.5.0
- spacy==3.7.4
- torch==2.1.1+cu118
- tqdm==4.66.2
- transformers==4.36.2

### Files

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
### Run our code

In the `code` directory:
- Run `chmod +x bash/*`.
- Train the initial model `bash/train_quad.sh -c 0 -d acos/rest16 -b quad -s 42`.
- Pseudo label with `bash/pseudo_labeling.sh -c 0 -d acos/rest16 -b quad`.
- Train the scorer `bash/train_scorer.sh -c 0 -d acos/rest16 -b scorer -s 42 -l 20 -t 01234+ -a 1`.
- Filter the pseudo-labeled data `bash/do_filtering.sh -c 0 -d acos/rest16 -b scorer`.

This will produce the filtered pseudo-labeled data, which can then be used to train the ASQP model:
- Train the ASQP model with pseudo-labeled data `bash/train_quad.sh -c 0 -d acos/rest16 -b 10-40_10000 -f 10-40_10000 -t ../output/filter/asqp/rest15.json`.
- Re-rank with `bash/do_reranking.sh -c 0 -d acos/rest16 -b scorer -q 10-40_10000 -a 2024-6-21`.

Note:
- The scorer significantly impacts final performance; adjusting its training parameters (lr, alpha, batch_size) is necessary.
- Before pseudo labeling, you need to unzip the files under `/code/data/raw`.
- Already filtered pseudo-labeled data is located in `/code/data/self_training`.

## Citation

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
