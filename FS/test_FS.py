

'''

adopt some algorithm for Feature Selection


'''




import DataStream as DS
import minepy as mp

from scipy.stats import pearsonr



def pearson(X, y):
    scores = []
    pvalues = []
    for i in range(X.shape[1]):
        score, pvalue = pearsonr(X[:, i], y)
        scores += [score,]
        pvalues += [pvalue,]
    return scores, pvalues
















