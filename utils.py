import numpy as np

def uncertainty(label_probs,n_points,measure):
    if measure=='least_confident':
        return np.argsort([1-max(y) for y in label_probs])[-n_points:]
    if measure=='margin':
        return np.argsort([np.sort(y)[-1]-np.sort(y)[-2] for y in label_probs])[:n_points]
    if measure=='entropy':
        return np.argsort([-sum(y*np.log(y)) for y in label_probs])[-n_points:]


def train_iteratively(data, model, measure, addn = 1, iterations = 25):
    Xtrain = data["train"]["X"]
    ytrain = data["train"]["y"]
    ninit = Xtrain.shape[0]

    Xpool = data["pool"]["X"]
    ypool = data["pool"]["y"]

    Xtest = data["test"]["X"]
    ytest = data["test"]["y"]


    #initial training set
    trainset= np.array([], dtype=int)
    poolidx=np.arange(len(Xpool),dtype=int)

    testacc = []
    testacc_byclass = [[] for i in range(7)]

    data = Xtrain
    labels = ytrain

    for i in range(iterations):
        model.fit(data, labels);
        #predict and calculate the accuracy
        ypred = model.predict(Xtest)

        #calculate accuracy on test set
        for j,clas in enumerate(sorted(set(ytest))):
            class_idx = ytest == clas
            testacc_byclass[j].append((ninit+i*addn, sum(ypred[class_idx]==clas)/sum(class_idx)))
        
        accuracy = sum(ytest == ypred)/len(ytest)
        testacc.append((ninit+i*addn,accuracy)) #add in the accuracy
        

        #find next index from pool (with highest uncertainty)
        pool = np.take(Xpool,poolidx,axis=0)
        prob = model.predict_proba(pool)

        if measure == "random":
            top_indices = np.random.choice(poolidx, addn)
        else:
            top_indices = poolidx[uncertainty(prob,addn,measure)]
            
        #update trainset and pool
        trainset = np.append(trainset, top_indices)
        poolidx = np.setdiff1d(poolidx,trainset)
        
        #add selected samples to training data
        data = np.vstack((data, np.take(Xpool, top_indices, axis=0)))
        labels = np.concatenate((labels, np.take(ypool, top_indices, axis=0)))

    return testacc, testacc_byclass