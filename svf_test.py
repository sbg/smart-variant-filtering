"""
Usage:
    svf_test.py (--table STR) [--alg STR] [--param STR] [--verbose]

Description:
    Smart Variant Filtration (SVF) test

Arguments:
    --table STR             Table with categorized variants used for learning
    --alg STR               Algorithm to test (ADA, KNN, SVM, RF, QDA, MLP [default: ADA]

Options:
    -h, --help                      Show this help message and exit.
    -v, --version                   Show version and exit.
    --verbose                       Log output
    --param STR                     Classifier parameters

Examples:
    python svf_test.py --table <raw_table> --alg KNN

"""

# Load libraries
import pandas
from sklearn import model_selection
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.model_selection import ShuffleSplit
#from imblearn.over_sampling import SMOTE
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import normalize
import operator
from docopt import docopt

def build_AdaBoostClassifier(alg, param = None):
    models = []

    # n_estimators : integer, optional (default=50)
    # The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
    n_estimators = [150, 200, 300, 400, 500, 1000]#[30, 50, 70, 100, 120]
    # learning_rate : float, optional (default=1.)
    # Learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learning_rate and n_estimators.
    learning_rate = [0.5, 1.0, 1.2, 1.5, 1.7]
    algorithm = ['SAMME', 'SAMME.R']
    if param:
        n_estimators = [int(param.split('-')[1])]
        learning_rate = [float(param.split('-')[2])]
        algorithm = [param.split('-')[3]]
    for n_e in n_estimators:
        for l_r in learning_rate:
            for a in algorithm:
                class_name = alg + '\t' + str(n_e) + '-' + str(l_r) + '-' + a
                models.append((class_name, AdaBoostClassifier(n_estimators=n_e, learning_rate=l_r, algorithm=a)))
                print class_name
    return models

def build_KNeighborsClassifier(alg, param = None):
    models = []
    neighbors = filter(lambda x: x % 4 != 0, list(range(1,50)))
    neighbors = [7, 9, 10, 13, 14, 15, 17, 19, 22, 25, 29, 30, 31, 33, 34, 35, 38, 39, 41, 43, 45, 49, 53]
    algorithms = ['ball_tree', 'kd_tree']
    p_distance = [1, 2]
    if param:
        neighbors = [int(param.split('-')[1])]
        algorithms = [param.split('-')[2]]
        p_distance = [int(param.split('-')[3])]
    for k in neighbors:
        for a in algorithms:
            for p in p_distance:
                class_name = alg + '\t' + str(k) + '-' + str(a) + '-' + str(p)
                models.append((class_name,KNeighborsClassifier(n_neighbors=k, algorithm=a, p=p, n_jobs=2)))
                print class_name
    return models

def build_SVM(alg, param = None):
    models = []
    C = [0.2, 0.6, 1.0, 1.5, 2.0, 4.0, 10.0, 100.0, 1000.0]
    kernels = ['linear', 'rbf']
    if param:
        C = [float(param.split('-')[1])]
        kernels = [param.split('-')[2]]
    for c in C:
        for k in kernels:
            class_name = alg + '\t' + str(c) + '-' + str(k)
            models.append((class_name,SVC(C=c, kernel=k)))
            print class_name
    return models

def build_RandomForestClassifier(alg, param = None):
    models = []
    n_estimators = [5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]
    criterion = ['gini', 'entropy']
    if param:
        n_estimators = [int(param.split('-')[1])]
        criterion = [param.split('-')[2]]
    for ne in n_estimators:
        for c in criterion:
            class_name = alg + '\t' + str(ne) + '-' + str(c)
            models.append((class_name, RandomForestClassifier(n_estimators=ne, criterion=c, n_jobs=2)))
            print class_name
    return models

def build_QuadraticDiscriminantAnalysis(alg, param = None):
    models = []
    tol = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    if param:
        tol = [float(param.split('-')[1])]
    for t in tol:
        class_name = alg + '\t' + str(t)
        models.append((class_name, QuadraticDiscriminantAnalysis(tol=t)))
        print class_name
    return models

def build_MLPClassifier(alg, param = None):
    models = []
    hidden_layer_sizes = [50, 100, 150, 250, 500, (10,10), (20,25), (30,50), (10,10,10), (20,30,40), (30,50,70)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.00001, 0.0001, 0.001]

    if param:
        hidden_layer_sizes_arr = (param.split('-')[1])
        hidden_layer_sizes = [tuple(map(int, hidden_layer_sizes_arr.split(',')))]
        activation = [param.split('-')[2]]
        solver = [param.split('-')[3]]
    for h in hidden_layer_sizes:
        for a in activation:
            for s in solver:
                class_name = alg + '\t' + str(h).strip(')').replace(' ','').strip('(') + '-'+str(a)+'-'+str(s)
                models.append(((class_name, MLPClassifier(hidden_layer_sizes=h, activation=a, solver=s))))
                print class_name
    return models

def build_model(alg, param = None):
    switcher = {
        'ADA': build_AdaBoostClassifier,
        'KNN': build_KNeighborsClassifier,
        'SVM': build_SVM,
        'RF': build_RandomForestClassifier,
        'QDA': build_QuadraticDiscriminantAnalysis,
        'MLP': build_MLPClassifier,
    }
    # Get the function from switcher dictionary
    func = switcher.get(alg)
    # Execute the function
    return func(alg, param)

args = docopt(__doc__, version='1.0')
verbose = args['--verbose']

# Read training data
variant_table = args['--table']
param = args['--param']

# Load dataset
#variant_table = 'data/wgs/HG001_50x_wgs_indels.table'
#variant_table = 'data/HG001_NIST7035.rep.indels.table'
df = pandas.read_csv(variant_table, sep = '\t')
df.__delitem__('CHROM')
df.__delitem__('POS')
df = df.fillna(0.)

# Split-out validation df
mask = df.iloc[:,-1] == 'TP'
df.loc[mask, df.columns[-1]] = int(1)
mask = ~mask
df.loc[mask, df.columns[-1]] = int(0)
array = df.values
X = array[:,0:7]
Y = array[:,8]
validation_size = 0.
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'f1'

if param:
    alg = param.split('-')[0] # Override alg setting
else:
    alg = args['--alg']
models = build_model(alg, param) # Build classifier model

filename = variant_table.split('/').pop()
basename = '.'.join(filename.split('.')[0:-1])
out_name = alg + '_' + basename + '.tsv'
f = open(out_name, 'w')
f_sort = open('sorted_' + out_name, 'w')
header = 'Algorithm\tParameters\tStd\tMean'
f.write(header + '\n')
f_sort.write(header + '\n')
# evaluate each model in turn
results = []
names = []
stats = {}

kfold = model_selection.KFold(n_splits=10, random_state=seed)
for name, model in models:
    #kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train.astype(int), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = name + '\t' + "{:.4f}".format(cv_results.std())
    stats[msg] = "{:.5f}".format(cv_results.mean())
    msg = msg + '\t' + "{:.5f}".format(cv_results.mean())
    print(msg)
    f.write(msg + '\n')

msg = ('#'*15 + ' S O R T E D ' + '#'*15)
print msg
stats = sorted(stats.items(), key=operator.itemgetter(1))
# Compare Algorithms
for it in stats:
    msg = str(it[0] + '\t' + it[1]).replace('\\t', '\t')
    print msg
    f_sort.write(msg + '\n')
f.close()
f_sort.close()



# Copyright 2018 Seven Bridges Genomics Inc. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
