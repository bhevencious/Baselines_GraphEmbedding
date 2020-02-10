from __future__ import print_function


import numpy, time
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=False)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y, log_key="undefined"):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y_ = numpy.asarray(numpy.round(Y_, decimals=0), dtype=numpy.int32)
        Y = self.binarizer.transform(Y)
        '''
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        '''
        log_file = open("eval_log.txt", "a")
        print('-------------------')
        print("class \t accuracy \t roc_score")
        print("class \t accuracy \t roc_score", file=log_file)
        for p in range(Y.shape[1]):
            acc_res = accuracy_score(Y[:,p], Y_[:,p])
            roc_res = roc_auc_score(Y[:,p], Y_[:,p])
            print("%d: \t %.2f \t\t %.2f" % (p, acc_res, roc_res))
            print("%d: \t %.2f \t\t %.2f" % (p, acc_res, roc_res), file=log_file)
        results = classification_report(Y, Y_)
        print(results)
        print(results, file=log_file)
        print('-------------------')
        log_file.close() 
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, X_train, Y_train, X_test, Y_test, log_key="undefined"):
        start_time = time.time()  # START: Training Time Tracker
        seed = 0
        numpy.random.seed(seed)
        state = numpy.random.get_state()       

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        end_time = time.time()  # STOP: Training Time Tracker
        
        log_file = open("eval_log.txt", "a")
        print("\n\n----"+log_key+"----")
        print("\n\n----"+log_key+"----", file=log_file)
        print("\nTraining Time:", end_time - start_time, "seconds\n")  # PRINT: Training Time Tracker
        print("Training Time:", end_time - start_time, "seconds", file=log_file)
        log_file.close()
        
        return self.evaluate(X_test, Y_test, log_key)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    if skip_head:
        fin.readline()
    while 1:        
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


# CUSTOM: Format date in "shampoo_sales" dataset (used in "load_data()" function)
def custom_date_parser(self, raw_date):
    return datetime.strptime(raw_date, '%Y %m %d %H')


# Load data from LOCAL directory after REMOTE extraction    
def load_data(local_path, file_name, sep="\s", header=0, index_col=0, mode="EXTRACT"):
    local_file = local_path + file_name
    if (file_name[-5:] == ".xlsx") or (file_name[-4:] == ".xls"):
        if (mode == "EXTRACT"):
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=custom_date_parser)
        elif (mode == "GRAPH"):
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
        else:
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
    elif (file_name[-4:] == ".csv"):
        if (mode == "EXTRACT"):
            return pd.read_csv(local_file, header=header, index_col=index_col, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=custom_date_parser)
        elif (mode == "GRAPH"):
            return pd.read_csv(local_file, header=header, index_col=index_col)
        else:
            return pd.read_csv(local_file, header=header, index_col=index_col)        
    else:
        return pd.read_table(local_file, sep=sep, header=header, index_col=index_col, engine='python')