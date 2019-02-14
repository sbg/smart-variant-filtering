"""
Usage:
    svf_train.py [--table_indel STR] [--table_snv STR] [--vcf STR] [--table_val STR] [--alg_param_indel STR] [--alg_param_snv STR] [--features_snv STR] [--features_indel STR] [--verbose]

Description:
    Train a Smart Variant Filtration (SVF) model that will be used for filtering of VCF file

Arguments:
    --table_indel STR         Table with indel categorized variants used for learning
    --table_snv STR           Table with SNV categorized variants used for learning
    --vcf STR                 Input VCF file
    --table_val STR           Table for validation when applying model
    --alg_param_indel STR     Comma separated list of algorithm and its parameters [default: ADA,150,1.0,SAMME]
    --alg_param_snv STR       Comma separated list of algorithm and its parameters [default: ADA,150,1.0,SAMME]
    --features_snv STR        Comma separated list of features used for SNVs [default: QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR]
    --features_indel STR      Comma separated list of features used for indels [default: QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR]

Options:
    -h, --help                      Show this help message and exit.
    -v, --version                   Show version and exit.
    --verbose                       Log output

Examples:
    python svf_train.py --table_indel data/wes/6_features/HG001_NIST7035_dbsnp_indels.table --table_snv data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.table --alg_param_indel MLP,250,logistic,sgd --alg_param_snv MLP,500,tanh,adam --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf
    python svf_train.py --table_indel data/wes/7_features/HG002_oslo_exome_dbsnp_indels.table --table_snv data/wes/7_features/HG002_oslo_exome_dbsnp_SNVs.table --alg_param_indel MLP,10,logistic,sgd --alg_param_snv MLP,10,logistic,sgd --vcf data/wes/7_features/HG001_NIST7035_raw.dbsnp.vcf --features_snv QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR,dbSNPBuildID --features_indel QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR,dbSNPBuildID

"""

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle
import sys
from docopt import docopt
from sklearn.preprocessing import StandardScaler
if sys.version_info > (3, 0):
    from past.builtins import execfile

def build_model(alg_param):
    alg = alg_param.split(',')[0]
    param = alg_param.split(',')[1:]
    if alg == 'ADA': model = AdaBoostClassifier(n_estimators=int(param[0]), learning_rate=float(param[1]), algorithm=param[2])
    elif alg == 'KNN': model = KNeighborsClassifier(n_neighbors=int(param[0]), algorithm=param[1], p=float(param[2]))
    elif alg == 'RF': model = RandomForestClassifier(n_estimators=int(param[0]), criterion=param[1])
    elif alg == 'SVM': model = SVC(C=float(param[0]), kernel=param[1])
    elif alg == 'QDA': model = QuadraticDiscriminantAnalysis(tol=float(param[0]))
    elif alg == 'MLP': model = MLPClassifier(hidden_layer_sizes=int(param[0]), activation=param[1], solver=param[2])
    else: model = []

    return model


args = docopt(__doc__, version='1.0')
verbose = args['--verbose']

variant_tables = []
features_list = []
if args['--table_snv']:
    variant_tables.append(args['--table_snv'])
    features_list.append(args['--features_snv'])
if args['--table_indel']:
    variant_tables.append(args['--table_indel'])
    features_list.append(args['--features_indel'])
alg_params = [args['--alg_param_snv'], args['--alg_param_indel']]
multiple_val_fix = lambda x: float(x.split(',')[1]) if ',' in str(x) else float(x)

for i in range(0, len(variant_tables)):
    variant_table = variant_tables[i]
    alg_param = alg_params[i]
    df = pandas.read_csv(variant_table, sep='\t', dtype=str)

    #assert(features_list in df.columns)
    Y = df.iloc[:,-1]
    df = df[features_list[i].split(',')]
    df = df.fillna(0.)
    num_features = len(features_list[i].split(','))
    # array = df.values
    # X = df #array[:,0:num_features]
    X = df.applymap(multiple_val_fix) # map(multiple_val_fix, df)

    if 'dbSNPBuildID' in X.columns:
        X['dbSNPBuildID'] = X['dbSNPBuildID'].map(lambda x: 1.0 if x > 0 else 0.)
    # for col_num in range(num_features):
    #     if X.columns[col_num] == 'dbSNPBuildID':
    #         X[:, col_num][X[:, col_num] > 0] = 1.0  # Take into account only dbSNP membersip, does not metter which version
        #elif df.columns[col_num] == 'AF':
        #    X[:, col_num] = map(multiple_val_fix, X[:, col_num])


    validation_size = 0.0
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Make predictions on validation dataset
    model = build_model(alg_param)
    model.fit(X_train, Y_train)

    #Save model to file, parse file input name and create output name
    filename = variant_table.split('/').pop()
    basename = '.'.join(filename.split('.')[0:-1])
    if variant_table == args['--table_indel']:
        fname_model = basename + '_' + str(num_features) + '_features_indel.sav'
        fname_model_indel = fname_model
    else:
        fname_model = basename + '_' + str(num_features) + '_features_snv.sav'
        fname_model_snv = fname_model
    pickle.dump(model, open(fname_model, 'wb'))

if args['--vcf']:
    if args['--table_val']:
        sys.argv = ['svf_apply', '--vcf', args['--vcf'], '--table', args['--table_val'],
                '--indel_model', fname_model_indel, '--snv_model', fname_model_snv, '--verbose',
                '--features_snv', args['--features_snv'], '--features_indel', args['--features_indel'],
                    '--discard_existing_filters']
    else:
        sys.argv = ['svf_apply', '--vcf', args['--vcf'],
                '--indel_model', fname_model_indel, '--snv_model', fname_model_snv, '--verbose',
                '--features_snv', args['--features_snv'], '--features_indel', args['--features_indel'],
                '--discard_existing_filters']
    execfile('svf_apply.py')


# Copyright 2018 Seven Bridges Genomics Inc. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
