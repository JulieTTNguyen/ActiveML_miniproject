import numpy as np

def uncertainty(label_probs,n_points,measure):
    if measure=='least confident':
        return np.argsort([1-max(y) for y in label_probs])[-n_points:]
    if measure=='margin':
        return np.argsort([np.sort(y)[-1]-np.sort(y)[-2] for y in label_probs])[:n_points]
    if measure=='entropy':
        return np.argsort([-sum(y*np.log(y)) for y in label_probs])[-n_points:]


def train_iteratively(data, model, measure, ninit = 20, addn = 1):
    Xtrain = data["train"]["X"]
    ytrain = data["train"]["y"]

    Xpool = data["pool"]["X"]
    ypool = data["pool"]["y"]

    Xtest = data["test"]["X"]
    ytest = data["test"]["y"]
    
    #initial training set
    trainset= np.array([])
    poolidx=np.arange(len(Xpool),dtype=int)

    testacc = []

    for i in range(25):
        data = np.vstack((Xtrain,np.take(Xpool,trainset,axis=0)))
        labels = np.vstack((ytrain,np.take(ypool,trainset,axis=0)))
        model.fit(data, labels)
        #predict and calculate the accuracy
        ypred = model.predict(Xtest)

        #calculate accuracy on test set
        accuracy = sum(ytest == ypred)/len(ytest)
        testacc.append((ninit+i*addn,accuracy)) #add in the accuracy
        print('Model: LR, %i random samples'%(ninit+i*addn))

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