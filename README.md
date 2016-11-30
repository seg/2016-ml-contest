# 2016-ml-contest

Welcome to the *Geophysical Tutorial* Machine Learning Contest 2016! Read all about the contest in [the October 2016 issue](http://library.seg.org/toc/leedff/35/10) of the magazine. Look for Brendon Hall's tutorial on lithology prediction with machine learning.

**You can run the notebooks in this repo in the cloud, just click the badge below:**

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/seg/2016-ml-contest)

You can also clone or download this repo with the green button above, or just read the documents:

- [index.ipynb](index.ipynb) &mdash; All about the contest.
- [Facies_classification.ipynb](Facies_classification.ipynb) &mdash; Brendon's notebook with all the code you need to get started in machine learning in Python.


## Leaderboard

F1 scores of models against secret blind data in the STUART and CRAWFORD wells. The logs for those wells are available [in the repo](https://github.com/seg/2016-ml-contest/blob/master/validation_data_nofacies.csv), but contestants do not have access to the facies.

|   | Team                                          | F1         | Algorithm  | Language | Solution                 |
|:-:|-----------------------------------------------|:----------:|------------|----------|--------------------------|
| 1 | [gganssle](https://github.com/gganssle)       | **0.561**  | DNN        | Lua      |  [Notebook](gram/faye.ipynb) |
| 2 | MandMs                                        | **0.536**  | SVM        | Python      |  [Notebook](MandMs/Facies_classification-M%26Ms_SVM_rbf_kernel.ipynb) |
| 3 | LA_Team                                       | **0.535**<sup>1</sup>| DNN | Python |  [Notebook](LA_Team/Facies_classification_LA_TEAM_02.ipynb) |
| 4 | LA_Team                                       | **0.519**<sup>1</sup>| Boosted trees | Python |  [Notebook](LA_Team/Facies_classification_LA_TEAM.ipynb) |
| 5 | [CannedGeo](https://github.com/cannedgeo)     | **0.512**  | SVM        | Python | [Notebook](CannedGeo_/Facies_classification-BPage_CannedGeo_F1_56-VALIDATED.ipynb) |
| 6 | [BrendonHall](https://github.com/brendonhall) | **0.412**  | SVM        | Python | Initial score in article |

<sup>1</sup>&nbsp;Pending complete validation. This usually takes us a few days.

## Getting started with Python

Please refer to the [User guide to the geophysical tutorials](http://library.seg.org/doi/abs/10.1190/tle35020190.1) for tips on getting started in Python and find out more about Jupyter notebooks.


## Find out more about the contest

If you intend to enter this contest, I suggest you check [the open issues](https://github.com/seg/2016-ml-contest/issues) and read through  [the closed issues](https://github.com/seg/2016-ml-contest/issues?q=is%3Aissue+is%3Aclosed) too. There's some good info in there.

To find out more please read the article in [the October issue](http://library.seg.org/toc/leedff/35/10) or read the manuscript in the [`tutorials-2016`](https://github.com/seg/tutorials-2016) repo.


## Rules

We've never done anything like this before, so there's a good chance these rules will become clearer as we go. We aim to be fair at all times, and reserve the right to make judgment calls for dealing with unforeseen circumstances.

**IMPORTANT: When this contest was first published, we asked you to hold the SHANKLE well blind. This is no longer necessary. You can use all the published wells in your training. Related: I am removing the file of predicted facies for the STUART and CRAWFORD wells, to reduce confusion — they are not actual facies, only those predicted by Brendon's first model.**

- You must submit your result as code and we must be able to run your code.
- **Entries will be scored by a comparison against known facies in the STUART and CRAWFORD wells, which do not have labels in the contest dataset. We will use the F1 cross-validation score.** See [issue #2 regarding this point.](https://github.com/seg/2016-ml-contest/issues/2). The scores in the 'leaderboard' reflect this.
- The result we get with your code is the one that counts as your result.
- To make it more likely that we can run it, your code must be written in Python or R or Julia **or Lua** [updated 26 Oct].
- The contest is over at 23:59:59 UT (i.e. midnight in London, UK) on 31 January 2017. Pull requests made aftetr that time won't be eligible for the contest.
- If you can do even better with code you don't wish to share fully, that's really cool, nice work! But you can't enter it for the contest. We invite you to share your result through your blog or other channels... maybe a paper in *The Leading Edge*.
- This document and documents it links to will be the channel for communication of the leading solution and everything else about the contest.
- This document contains the rules. Our decision is final. No purchase necessary. Please exploit artificial intelligence responsibly. 

## Licenses

**Please note that the dataset is not openly licensed. We are working on this, but for now please treat it as proprietary. It is shared here exclusively for use on this problem, in this contest. We hope to have news about this in early 2017, if not before.**

All code is the property of its author and subject to the terms of their choosing. If in doubt — ask them.

The information about the contest, and the original article, and everything in this repo published under the auspices of SEG, is licensed CC-BY and OK to use with attribution.
