

'''

adopt some algorithm for Feature Selection


'''




import DataStream as DS
import numpy as np
import pandas as pd
from minepy import MINE

from scipy.stats import pearsonr


# Pearson Correlation
# scores: linear correlation [-1,1]
# pvalues:
def pearson(X, y):
    scores = []
    pvalues = []
    for i in range(X.shape[1]):
        score, pvalue = pearsonr(X[:, i], y)
        scores += [score,]
        pvalues += [pvalue,]
    return scores, pvalues


def MIC(X, y):
    mics = []
    for i in range(X.shape[1]):
        m = MINE()
        m.compute_score(X[:, i], y)
        mic = m.mic()
        mics += [mic,]
    return mics

def MIC_selected(featureName, mics, top=10):
    scores = np.array(mics)
    features = zip(featureName, scores)
    features = sorted(features, key=lambda tup: tup[1], reverse=True)
    set_selected = []
    for i in range(top):
        set_selected += [features[i][0],]
    return set_selected

# word count return by the form of (dictionary --> list)
# ...
def Counter(words):
    dictionary = {}
    for i in range(words.shape[0]):
        for j in range(words.shape[1]):
            word = words[i, j]
            dictionary[word] = 0
    for i in range(words.shape[0]):
        for j in range(words.shape[1]):
            word = words[i, j]
            dictionary[word] += 1
    dictionary = sorted(zip(dictionary.keys(), dictionary.values()), key=lambda tup: tup[1], reverse=True)
    dictionary = dictionary[:Top]
    features = []
    counts = []
    for i in range(len(dictionary)):
        features += [dictionary[i][0],]
        counts += [dictionary[i][1],]
    return features, counts



PATH = '../FS/csv/'
Top = 10

if __name__ == "__main__":

    FileNames = DS.fetch_FileNames(PATH)
    FileNames.sort()  # actually not in normal order
    print("There are totally " + str(len(FileNames)) + " csv files.")

    #data, LoadScore, PerScore, snapId, featureName = DS.DataPrepared(FileNames[3])
    #mics = MIC(data, PerScore)

    #set_selected = MIC_selected(featureName, mics, Top)
    #print(set_selected)


    # use MIC FS-method and store the result as csv files
    #
    #
    it = len(FileNames)
    DBID = []
    subset = []
    for i in range(it):
        name = FileNames[i]
        DBID += [name,]
        data, LoadScore, PerfScore, snapId, featureName = DS.DataPrepared(name)

        # LoadScore
        mics_LS = MIC(data, LoadScore)
        subset_LS = MIC_selected(featureName, mics_LS, Top)

        # PerfScore
        mics_PS = MIC(data, PerfScore)
        subset_PS = MIC_selected(featureName, mics_PS, Top)

        subset.append([name] + subset_LS + subset_PS)
    MIC_col_name = []
    MIC_col_name += ['DBID',]
    for i in range(Top*2):
        rank = i % Top
        if i / Top < 1:
            MIC_col_name += ['LS-'+'Top'+str(rank),]
        else:
            MIC_col_name += ['PS-' + 'Top' + str(rank), ]

    MIC_col_name = np.array(MIC_col_name)
    DBID = np.array(DBID)
    #print(MIC_col_name)
    #print(DBID)
    subset = np.array(subset)
    #print(subset.shape)

    MIC_result = pd.DataFrame(subset, columns=MIC_col_name)
    #print(MIC_result)
    # MIC_result.to_csv('MIC_result.csv', encoding="utf-8")

    # MIC-method sum result of the 50 files
    words = MIC_result.values
    words = words[:, 1:]
    words_LS = words[:, 0:Top-1]
    words_PS = words[:, Top:]

    features_LS, counts_LS = Counter(words_LS)
    features_PS, counts_PS = Counter(words_PS)
    #print(features_LS)
    #print(counts_LS)

    MIC_result.loc['features'] = ['all DBs'] + features_LS + features_PS
    MIC_result.loc['counts'] = ['all DBs'] + counts_LS + counts_PS
    #print(MIC_result)
    MIC_result.to_csv('MIC_result.csv', encoding="utf-8")



















