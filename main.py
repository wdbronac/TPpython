from __future__ import division

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from math import *
#nsamples = 100
#a = datasets.make_moons(n_samples=nsamples, shuffle=True, noise=0.25, random_state=None)
#X = a [0]
#plt.scatter(X[0:,0],X[0:,1])
#plt.show()
plot_step = 0.02

#generating a test dataset

nsamples = 10000
test = datasets.make_moons(n_samples=nsamples, shuffle=True, noise=0.25, random_state=None)


#training on a dataset


nsamples = 100
data_train = datasets.make_moons(n_samples=nsamples, shuffle=True, noise=0.25, random_state=None)
train = data_train[0]
clf = DecisionTreeClassifier().fit(data_train[0],data_train[1], sample_weight=None, check_input=True)

#testing
risk =1/len(test[1])*np.sum(abs(test[1]-clf.predict(test[0])))

##plotting the decision boundary
#y_min, y_max = train[:, 1].min() - 1, train[:, 1].max() + 1
#x_min, x_max = train[:, 0].min() - 1, train[:, 0].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#np.arange(y_min, y_max, plot_step))
#
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#plt.show()

nb_moy = 30
risk_tot = np.array([])
risk_tot_var = np.array([])

for i in range(5,50):
    I = 10*i
    nsamples = I
    risk = np.array([])
    for j in range(nb_moy):
        data_train = datasets.make_moons(n_samples=nsamples, shuffle=True, noise=0.25, random_state=None)
        train = data_train[0]
        clf = DecisionTreeClassifier().fit(data_train[0],data_train[1], sample_weight=None, check_input=True)
        risk =np.append(risk,np.sum(abs(test[1]-clf.predict(test[0]))))
    risk = risk/len(test[1])
    risk_tot = np.append(risk_tot, np.mean(risk))
    risk_tot_var = np.append(risk_tot_var, np.std(risk))


vec = np.array(range(5,50))*10
figure(3)
plt.plot(vec, risk_tot)
figure(4)
plt.plot(vec, risk_tot_var)



#with ensemble

nb_moy = 50
risk_tot = np.array([])
risk_tot_var = np.array([])
nmax = 100
for i in range(5,nmax):
    I = 10*i
    nsamples = I
    risk = np.array([])
    pred = 0
    data_train = datasets.make_moons(n_samples=nsamples, shuffle=True, noise=0.25, random_state=None)
    train = data_train[0]
    for j in range(nb_moy):
        clf = DecisionTreeClassifier().fit(data_train[0],data_train[1], sample_weight=None, check_input=True)
        pred= pred+clf.predict(test[0])
    pred = pred/nb_moy #pred is a list of predictions: the means of every predictor
    pred = np.round(pred) #pred is a list of predictions: the VOTES of all the predictors
    
    risk =np.sum(abs(test[1]-pred)) 
    risk = risk/len(test[1]) #total risk of our ensemble predictor
    risk_tot = np.append(risk_tot, risk)


vec = np.array(range(5,nmax))*10



figure(1)
plt.plot(vec, risk_tot)
