# White Wine Quality Classification: Gaussian Bayes vs k-NN

A side-by-side comparison of two classical classifiers, a **Gaussian Bayesian classifier** and **k-nearest neighbours (k-NN)**, for predicting white-wine quality from physicochemical measurements.

The main finding is that **the model with the best accuracy is not the model with the best balanced performance**. k-NN reaches 51.7% accuracy, but the Gaussian Bayesian classifier has the best macro-F1 because it does better on the rare quality classes.

## Why I built this

Wine quality is scored by human tasters, and scores vary from person to person. Lab measurements such as acidity, sulphur dioxide and alcohol are objective, so it is tempting to predict quality from them. I wanted to see how far two simple, transparent methods could go, and what the results look like on a problem where some classes are extremely rare.

I implemented both classifiers myself from their mathematical definitions, which means the Gaussian model's priors, class means and class-specific covariance matrices are all estimated directly. The goal was a clear, honest **baseline** that more complex models can later be compared against.

## Key results

Evaluated on a held-out test set of 793 wines:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Gaussian Bayesian | 0.4754 | 0.3135 | **0.2925** | **0.2909** | 0.4695 |
| k-NN (Euclidean, k=1) | 0.4779 | 0.2598 | 0.2535 | 0.2558 | 0.4751 |
| k-NN (Euclidean, k=3) | 0.4830 | 0.2813 | 0.2501 | 0.2476 | 0.4761 |
| k-NN (Euclidean, k=5) | 0.4893 | 0.3041 | 0.2349 | 0.2432 | 0.4739 |
| k-NN (Euclidean, k=7) | 0.4994 | 0.3231 | 0.2349 | 0.2438 | 0.4822 |
| k-NN (Euclidean, k=9) | **0.5170** | 0.2994 | 0.2345 | 0.2415 | 0.4955 |
| k-NN (Manhattan, k=5) | **0.5170** | **0.3316** | 0.2415 | 0.2495 | **0.4977** |

**What the numbers say**

- **Accuracy and balance disagree.** As k grows from 1 to 9, accuracy rises (47.8% to 51.7%) while macro-F1 falls (0.256 to 0.242). Larger neighbourhoods get better at the common classes and worse at the rare ones.
- **Gaussian Bayes wins on balance.** It has the highest macro recall and macro F1, even though its accuracy is lower than most k-NN settings.
- **Manhattan beat Euclidean at k=5.** Accuracy went from 48.9% to 51.7% and macro-F1 from 0.243 to 0.250, though the gap is modest.
- **Rare classes are hard for both models.** For the Gaussian Bayesian classifier, quality 5 and 6 scored F1 of 0.53 and 0.49, while quality 8 scored 0.06 and quality 9 scored 0.

## The data

I used the white-wine file from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009).

- **4,898 samples**, 11 physicochemical features, and a quality score from 3 to 9
- **Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Duplicates:** 937 duplicate rows (19.13%) were removed, leaving **3,961 unique samples**. This matters because duplicates can land in both the training and test sets and inflate the scores.
- **Severe class imbalance:**

| Quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| Samples | 20 | 153 | 1,175 | 1,788 | 689 | 131 | 5 |

The largest class is about 358 times the size of the smallest. This is why the project reports macro-averaged metrics next to accuracy.

Some features are strongly related. Residual sugar and density are correlated at 0.82, and alcohol and density at -0.76, so the predictors overlap in the information they carry.

## Method

1. **Clean:** remove duplicate rows (3,961 remain).
2. **Split:** 80/20 into 3,168 training and 793 test samples.
3. **Scale:** standardise each feature using the **training set's mean and standard deviation only**, then apply the same transform to the test set, so no test information leaks into training.
4. **Gaussian Bayesian classifier:** for each quality class, estimate a prior, an 11-dimensional mean vector and a full 11x11 covariance matrix. Predict the class with the highest log-discriminant:

   `g_c(x) = -1/2 log|Σ_c| - 1/2 (x - μ_c)ᵀ Σ_c⁻¹ (x - μ_c) + log P(c)`

5. **k-NN:** classify by majority vote among the k nearest training samples, for k in {1, 3, 5, 7, 9} with Euclidean distance, then compare Euclidean and Manhattan distance at k = 5.
6. **Evaluate:** accuracy, plus precision, recall and F1 in both macro-averaged (every class counts equally) and weighted (classes count by size) forms.

## Limitations

- **One train/test split.** With 793 test samples and only small gaps between several models, a single split cannot say reliably which model is better. Repeated or stratified cross-validation would give firmer conclusions.
- **Minority classes are barely represented.** Quality 9 has only 5 samples in the whole dataset, so scores for the extreme classes are unreliable.
- **Narrow search.** I tried five values of k and two distance metrics, so these results do not establish a best k or best metric.
- **One dataset.** Findings may not carry over to other wine regions or datasets.
- **Classical baselines only.** I did not use resampling (such as SMOTE), feature engineering, ensembles or neural networks.

## Ideas for next steps

- Cross-validation with confidence intervals on each metric
- Class-balancing: resampling, class weights, or synthetic oversampling
- Feature selection, given the correlated predictors
- Stronger models (random forest, gradient boosting) compared against this baseline
- Treating quality as an ordered scale, since predicting 5 when the truth is 6 is a smaller error than predicting 9


## Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.

## Author

**Yaw Asante** - Data Scientist, Statistician and Financial Economist

Email: vkpoasante@gmail.com

## License
**MIT**



