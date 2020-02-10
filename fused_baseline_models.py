# Enforce CPU Usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
from ge.classify import read_node_label, load_data, Classifier
from ge import Node2Vec, SDNE, Struc2Vec, DeepWalk, LINE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings, X, Y, X_train, y_train, X_test, y_test, log_key):   
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='liblinear'))
    clf.split_train_evaluate(X, Y, X_train, y_train, X_test, y_test, log_key=log_key)

def plot_embeddings(embeddings, d_set):
    X, Y = read_node_label("data/"+d_set+"/"+d_set+".labels", skip_head=True)
    emb_list = []
    
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    
    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.legend()
    plt.show()


d_set = ["Facebook-Page2Page", "PubMed-Diabetes", "Terrorists-Relation"]  # ["Cora", "CiteSeer", "Facebook-Page2Page", "PubMed-Diabetes", "Terrorists-Relation", "Zachary-Karate", "Internet-Industry-Partnerships"]  # [sparse, dense]
mdl = ["Node2Vec", "SDNE", "DeepWalk", "LINE"]  # ["Node2Vec", "SDNE", "Struc2Vec", "DeepWalk", "LINE"]

for i in range(len(d_set)):
    # Load/Prepare data
    graph_fname = "data/"+d_set[i]+"/"+d_set[i]
    X, Y = read_node_label(graph_fname+".labels", skip_head=True)
    X = np.asarray(X)
    Y = np.asarray(Y)
     
    # Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strageies
    train_frac = 0.8
    test_frac = round((1 - train_frac), 1)
    print("Training classifier using {:.2f}% nodes...".format(train_frac * 100))
    if not os.path.isfile(graph_fname+"_strat_train_test.splits"):
        stratified_data = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, train_size=train_frac, random_state=42)
        for train_index, test_index in stratified_data.split(X, Y):
            strat_X_train, strat_y_train = X[train_index], Y[train_index]
            strat_X_test, strat_y_test = X[test_index], Y[test_index]
            # Preserve 'train' & 'test' stratified-shuffle-splits
            train_test_splits = pd.concat([pd.DataFrame(train_index), pd.DataFrame(test_index)], axis='columns', ignore_index=True)
            train_test_splits.to_csv(graph_fname+"_strat_train_test.splits", sep=" ", header=False, index=False)        
    else:
        strat_train_test = load_data(graph_fname, "_strat_train_test.splits", sep="\s", header=None, index_col=None, mode="READ")
        train_index, test_index = strat_train_test.values[:,0], strat_train_test.values[:,-1]  # "values()" method returns a NUMPY array wrt dataframes
        train_index, test_index = train_index[np.logical_not(np.isnan(train_index))], test_index[np.logical_not(np.isnan(test_index))]  # Remove nan values from arrays
        train_index, test_index = train_index.astype('int32'), test_index.astype('int32')
        strat_X_train, strat_y_train = X[train_index], Y[train_index]
        strat_X_test, strat_y_test = X[test_index], Y[test_index]
    # Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strageies
    
    for j in range(len(mdl)):
        if __name__ == "__main__":
            G = nx.read_edgelist("data/"+d_set[i]+"/"+d_set[i]+".edges", create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
            if (mdl[j] == "Node2Vec"):
                model = Node2Vec(G, walk_length = 10, num_walks = 80, p = 0.25, q = 4, workers = 1)
                model.train(window_size = 5, iter = 3)
            elif (mdl[j] == "SDNE"):	
                model = SDNE(G, hidden_size=[256, 128],)
                model.train(batch_size=3000, epochs=40, verbose=2)
            elif (mdl[j] == "Struc2Vec"):
                model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
                model.train(window_size = 5, iter = 3)
            elif (mdl[j] == "DeepWalk"):
                model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
                model.train(window_size=5, iter=3)
            elif (mdl[j] == "LINE"):
                model = LINE(G, embedding_size=128, order='second')
                model.train(batch_size=1024, epochs=50, verbose=2)
                
            embeddings = model.get_embeddings()	
        
            evaluate_embeddings(embeddings, X, Y, strat_X_train, strat_y_train, strat_X_test, strat_y_test, log_key=mdl[j]+": "+d_set[i])
            plot_embeddings(embeddings, d_set[i])