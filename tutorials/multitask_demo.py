
import os
import sys
sys.path.append('..')
import pickle
from metal.multitask import TaskHierarchy
from metal.multitask import MTLabelModel
from metal.multitask import MTEndModel
import torch

if __name__ == "__main__":

    with open("data/multitask_tutorial.pkl", 'rb') as f:
        #task num:3
        #LFs num:10
        #Xs:[(800,1000), (100,1000), (100,1000)];
        #Ys:[3*(800,), 3*(100,), 3*(100,)];
        #Ls:[3*sparse matrix(800,10),3*sparse matrix(100,10),3*sparse matrix(100,10)];
        #Ds:[800*str, 100*str, 100*str]
        Xs, Ys, Ls, Ds = pickle.load(f)
#         print("Ds[1]:{}".format(Ds[0]))
        print("size Xs:{}, size Ys:{}, size Ls:{}, size Ds:{}".format(len(Xs), len(Ys), len(Ls), len(Ds)))
        
        print("items Xs:{}, items Ys:{}, items Ls:{}, items Ds:{}".format(Xs[0].shape, Ys[0][0].shape, len(Ls[0]), len(Ds[0])))
#         print("val X[0]:{}, val X[1]:{}, val X[2]:{},".format(Xs[0][0], Xs[1][0], Xs[2][0]))
#         print("val Y[0]:{}, val Y[1]:{}, val Y[2]:{},".format(Ys[0][0], Ys[0][1], Ys[0][2]))
    task_graph = TaskHierarchy(cardinalities=[2,3,3], edges=[(0,1), (0,2)])
    label_model = MTLabelModel(task_graph=task_graph)
    
    label_model.train_model(Ls[0], n_epochs=200, log_train_every=20, seed=123)
    label_model.score((Ls[1], Ys[1]))
    # Y_train_ps stands for "Y[labels]_train[split]_p[redicted]s[oft]"
    Y_train_ps = label_model.predict_proba(Ls[0])
    
    
    
    end_model = MTEndModel([1000, 100, 10], task_graph=task_graph, seed=123)
    end_model.train_model((Xs[0], Y_train_ps), valid_data=(Xs[1], Ys[1]), n_epochs=5, seed=123)
    print("Label Model:")
    score = label_model.score((Ls[2], Ys[2]))
    print()
    
    print("End Model:")
    score = end_model.score((Xs[2], Ys[2]))
    print(score)
    
    scores = end_model.score((Xs[2], Ys[2]), reduce=None)
    print(scores)
    
    Y_p = end_model.predict(Xs[2])
    print(Y_p)
