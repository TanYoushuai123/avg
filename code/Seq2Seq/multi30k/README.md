---
task_categories:
- translation
language:
- en
- de
size_categories:
- 10K<n<100K
---
# Multi30k

This dataset contains the "multi30k" dataset, which is the "task 1" dataset from [here](https://www.statmt.org/wmt16/multimodal-task.html).

Each example consists of an "en" and a "de" feature. "en" is an English sentence, and "de" is the German translation of the English sentence.

### Data Splits

The Multi30k dataset has 3 splits: _train_, _validation_, and _test_.

| Dataset Split | Number of Instances in Split                |
| ------------- | ------------------------------------------- |
| Train         | 29,000                                      |
| Validation    | 1,014                                       |
| Test          | 1,000                                       |

### Citation Information

```
@article{elliott-EtAl:2016:VL16,
 author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
 title     = {Multi30K: Multilingual English-German Image Descriptions},
 booktitle = {Proceedings of the 5th Workshop on Vision and Language},
 year      = {2016},
 pages     = {70--74},
 year      = 2016
}
```