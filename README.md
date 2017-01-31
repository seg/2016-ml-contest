# 2016-ml-contest

Welcome to the *Geophysical Tutorial* Machine Learning Contest 2016! Read all about the contest in [the October 2016 issue](http://library.seg.org/toc/leedff/35/10) of the magazine. Look for Brendon Hall's tutorial on lithology prediction with machine learning.

**You can run the notebooks in this repo in the cloud, just click the badge below:**

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/seg/2016-ml-contest)

You can also clone or download this repo with the green button above, or just read the documents:

- [index.ipynb](index.ipynb) &mdash; All about the contest.
- [Facies_classification.ipynb](Facies_classification.ipynb) &mdash; Brendon's notebook with all the code you need to get started in machine learning in Python.


## Leaderboard

F1 scores of models against secret blind data in the STUART and CRAWFORD wells. The logs for those wells are available [in the repo](https://github.com/seg/2016-ml-contest/blob/master/validation_data_nofacies.csv), but contestants do not have access to the facies.

**Please note that after the contest closes, we will be applying a stochastic scoring approach to the leading models. So these scores are subject to change.**

| Team                                          | F1         | Algorithm     | Language | Solution                 |
|-----------------------------------------------|:----------:|---------------|----------|--------------------------|
| LA_Team                                       | **0.641**  | Boosted trees | Python   | [Notebook](LA_Team/Facies_classification_LA_TEAM_05.ipynb) |
| SHandPR                                       | **0.631**  | Boosted trees | Python   | [Notebook](SHandPR/Face_classification_SHPR_GradientBoost.ipynb) |
| [bestagini](https://github.com/bestagini)     | **0.631**  | Boosted trees | Python   | [Notebook](ispl/facies_classification_try03.ipynb) |
| HouMath                                       | **0.630**  | Boosted trees | Python   | [Notebook](HouMath/Face_classification_HouMath_XGB_06.ipynb) |
| [esaTeam](https://github.com/esa-as)          | **0.626**  | Boosted trees | Python   | [Notebook](esaTeam/esa_Submission01.ipynb) |
| Pet_Stromatolite                              | **0.625**  | Boosted trees | Python   | [Notebook](Pet_Stromatolite/Facies_Classification_Draft7.ipynb) |
| PA Team                                       | **0.623**  | Boosted trees | Python   | [Notebook](PA_Team/PA_Team_Submission_8_XGB.ipynb) |
| [ar4](https://github.com/ar4)                 | **0.606**  | Random forest | Python   | [Notebook](ar4/ar4_submission2.ipynb) |
| [Houston_J](https://github.com/Houston_J)     | **0.600**  | Boosted trees | Python   | [Notebook](Houston_J/Houston_J-sub2.ipynb) |
| Bird Team                                     | **0.598**  | Random forest | Python   | [Notebook](Bird_Team/Facies_classification_4.ipynb) |
| geoLEARN                                      | **0.594**  | Random forest | Python   | [Notebook](geoLEARN/Submission_3_RF_FE.ipynb) |
| [gccrowther](https://github.com/gccrowther)   | **0.589**  | Random forest | Python   | [Notebook](GCC_FaciesClassification/05%20-%20Facies%20Determination.ipynb) |
| [thanish](https://github.com/thanish)         | **0.580**  | Random forest | R        | [Code](Mendacium/Mendacium/rf_sub_10.R) |
| MandMs                                        | **0.579**  | Majority voting | Python | [Notebook](MandMs/02_Facies_classification-MandMs_plurality_voting_classifier.ipynb) |
| [evgenizer](https://github.com/evgenizer)     | **0.578**  | Boosted trees | Python   | [Notebook](EvgenyS/Facies_classification_ES.ipynb) |
| jpoirier                                      | **0.574**  | Random forest    | Python   | [Notebook](jpoirier/jpoirier011_submission001.ipynb) |
| [kr1m](https://github.com/kr1m)               | **0.570**  | AdaBoosted trees | Python   | [Notebook](Kr1m/Kr1m_SEG_ML_Attempt1.ipynb) |
| [ShiangYong](https://github.com/ShiangYong)   | **0.570**  | ConvNet          | Python   | [Notebook](ShiangYong/facies_classification_cnn.ipynb) |
| CarlosFuerte                                  | **0.570**  | Multilayer perceptron | Python      | [Notebook](CarlosFuerte/ML_Submission.ipynb) |
| [fvf1361](https://github.com/fvf1361)         | **0.568**  | Majority voting | Python   | [Notebook](fvf/facies_classification.ipynb) |
| [gganssle](https://github.com/gganssle)       | **0.561**  | Deep neural net | Lua      | [Notebook](gram/faye.ipynb) |
| [CarthyCraft](https://github.com/CarthyCraft) | **0.561**  | Boosted trees | Python | No code yet |
| StoDIG                                        | **0.561**  | ConvNet              | Python   | [Notebook](StoDIG/Facies_classification_StoDIG_4.ipynb) |
| [wouterk1MSS](https://github.com/wouterk1MSS) | **0.559**  | Random forest | Python   | [Notebook](MSS_Xmas_Trees/ml_seg_try1.ipynb) |
| [CEsprey](https://github.com/CEsprey)         | **0.550**  | Majority voting | Python | [Notebook](CEsprey%20-%20RandomForest/Facies_Tree_Ensemble_Classifier.ipynb) |
| [osorensen](https://github.com/osorensen)     | **0.549**  | Boosted trees | R        | [Notebook](boostedXmas/Facies%20Classification.ipynb) |
| [rkappius](https://github.com/rkappius)       | **0.534**  | Neural network           | Python   | [Notebook](rkappius/facies_w_tf_submit.py) |
| [JesperDramsch](https://github.com/JesperDramsch) | **0.530**  | Random forest | Python   | [Notebook](JesperDramsch/Facies_classification_Xmas_Trees-Copy1.ipynb) |
| BGC_Team                                      | **0.519**  | Deep neural network  | Python   | [Notebook](BGC_Team/Facies%20Prediction_submit.ipynb) |
| [CannedGeo](https://github.com/cannedgeo)     | **0.512**  | Support vector machine | Python   | [Notebook](CannedGeo_/Facies_classification-BPage_CannedGeo_F1_56-VALIDATED.ipynb) |
| ARANZGeo                                      | **0.511**  | Deep nerual network  | Python   | [Code](ARANZGeo/hypter.py) |
| [daghra](https://github.com/dagrha)           | **0.506**  | k-nearest neighbours  | Python   | [Notebook](dagrha/KNN_submission_1_dagrha.ipynb) |
| [cako](https://github.com/cako)               | **0.505**  | Multi-layer perceptron  | Python   | [Notebook](DiscerningHaggis/Discerning_Haggis_Facies_Classification.ipynb) |
| [BrendonHall](https://github.com/brendonhall) | **0.427**  | Support vector machine | Python   | Initial score in article |


## Getting started with Python

Please refer to the [User guide to the geophysical tutorials](http://library.seg.org/doi/abs/10.1190/tle35020190.1) for tips on getting started in Python and find out more about Jupyter notebooks.


## Find out more about the contest

If you intend to enter this contest, I suggest you check [the open issues](https://github.com/seg/2016-ml-contest/issues) and read through  [the closed issues](https://github.com/seg/2016-ml-contest/issues?q=is%3Aissue+is%3Aclosed) too. There's some good info in there.

To find out more please read the article in [the October issue](http://library.seg.org/toc/leedff/35/10) or read the manuscript in the [`tutorials-2016`](https://github.com/seg/tutorials-2016) repo.


## Rules

We've never done anything like this before, so there's a good chance these rules will become clearer as we go. We aim to be fair at all times, and reserve the right to make judgment calls for dealing with unforeseen circumstances.

**IMPORTANT: When this contest was first published, we asked you to hold the SHANKLE well blind. This is no longer necessary. You can use all the published wells in your training. Related: I am removing the file of predicted facies for the STUART and CRAWFORD wells, to reduce confusion — they are not actual facies, only those predicted by Brendon's first model.**

- You must submit your result as code and we must be able to run your code.
- **Entries will be scored by a comparison against known facies in the STUART and CRAWFORD wells, which do not have labels in the contest dataset. We will use the F1 cross-validation score.** See [issue #2 regarding this point](https://github.com/seg/2016-ml-contest/issues/2). The scores in the 'leaderboard' reflect this.
- Where there is stochastic variance in the predictions, the median average of 100 realizations will be used as the cross-validation score. See [issue #114 regarding this point](https://github.com/seg/2016-ml-contest/issues/114). The scores in the leaderboard **do not** currently reflect this. Probably only the top entries will be scored in this way. [updated 23 Jan]
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
