# About

DRR with encoder/decoder type model

# CoNLL
- About the [task](http://www.cs.brandeis.edu/~clp/conll16st/intro.html)
- About the [dataset](http://www.cs.brandeis.edu/~clp/conll16st/dataset.html)
- [tutorial](http://nbviewer.jupyter.org/github/attapol/conll16st/blob/master/tutorial/tutorial.ipynb) on the data
- 2016 task results [review][conll]

# Dataset breakdown

**The Pitler et al 2009 breakdown:**

| Set         | WSJ sections       |
|-------------|--------------------|
| Training    | 2-20               |
| Development | 0-1, can use 23-24 |
| Test        | 21-22              |

Followed by, for example: [Zhang et al 2015], [Chen et al, 2016], [Ji and Eisensteing, 2015]


**The CoNLL breakdown, recommended by the original PDTB 2.0 corpus:**

| Set         | WSJ sections |
|-------------|--------------|
| Training    | 2-21         |
| Development | 22           |
| Test        | 23           |

Followed by [CoNLL](http://www.aclweb.org/anthology/K/K16/K16-2.pdf#page=11), [Wang and Lan, 2016](https://www.aclweb.org/anthology/K/K16/K16-2.pdf#page=43)

# Relations
[CoNLL][conll] version classifies the lower 16 levels, and includes EntRel.

### 1st level, one-v-all
For higher level classification, such as in [Chen et al, 2016], they experiment with one-v-all with negative sampling from section 2-20. They use the Pitler breakdown and merge EntRel with Expansion.

[Chen et al, 2016]: https://www.aclweb.org/anthology/P/P16/P16-1163.pdf
[Zhang et al 2015]: http://www.anthology.aclweb.org/D/D15/D15-1266.pdf
[conll]: http://www.aclweb.org/anthology/K/K16/K16-2.pdf#page=26
[Ji and Eisensteing, 2015]: https://arxiv.org/pdf/1411.6699.pdf
