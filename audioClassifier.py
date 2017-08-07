#! python3
#
#   audioClassifier.py - takes in audio files, gets them into the proper format,
#                       renames them, performs feature extraction, and trains classifiers
#
#   Note: Read Audio Event Detection wiki page for complete info. 
#
#   renameFiles() - After splitting the files on labels using wavesurfer, will have
#                   extra info in the file name. This takes that away.
#   determineFeatures() - returns a 23 feature long vector for each class example located
#                       in the given directory. Saves it to a python file with the class name.
#   trainClassifier() - extracts the feature vectors from the given file, splits the data, and
#                       trains multiple classifiers. Then implements a random forest to determine
#                       the most relevant features (not completely implemented).
import os
import pickle
import random
import numpy as np
import pprint
import audioTrainTest as aT
import audioBasicIO
import audioFeatureExtraction
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
import mic
from sklearn import svm
from sklearn import tree
from matplotlib.colors import ListedColormap
import math
import logging
import pydotplus

filePath = 'C:\\Users\\...'
classifyPath = 'C:\\Users\\...'

def renameFiles():

    carC=0
    junkC=0
    martaC=0

    os.chdir(filePath)

    if not os.path.exists(filePath+'car'):
        os.mkdir(filePath+'\\car')
    if not os.path.exists(filePath+'junk'):
        os.mkdir(filePath+'\\junk')
    if not os.path.exists(filePath+'marta'):
        os.mkdir(filePath+'\\marta')
    for filename in os.listdir(filePath):
        
        print('Renaming: "%s"' % filename)
        if filename[-7:] == 'car.wav':
            newName = 'car'+str(carC)+'.wav'
            os.rename(filename, 'car\\'+newName)
            carC += 1
        elif filename[-8:] == 'junk.wav':
            newName = 'junk'+str(junkC)+'.wav'
            os.rename(filename, 'junk\\'+newName)
            junkC += 1
        elif filename[-9:] == 'marta.wav':
            newName = 'marta'+str(martaC)+'.wav'
            os.rename(filename, 'marta\\'+newName)
            martaC += 1
            
def determineFeatures(className):
    dataFile = className+'Features.py'
    X = []
    y = []
    try:
        for fileName in os.listdir(classifyPath+className):
            frames, fs = mic.getRecordedFrames(classifyPath+className+"\\"+fileName)
            feature = calc_features(frames,fs)

            print('Working on file: "%s"' % fileName)
            X.append(feature)
            y.append(className)

        fileObj=open(dataFile,'a')
        fileObj.write('X = ' + pprint.pformat(X) +'\n')
        fileObj.write('y = ' + pprint.pformat(y) +'\n')
        fileObj.close()

    except:
        print("The data broke it")

    
def trainClassifier():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    from itertools import product
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    #This is whatever you saved your X and y data into
    import carAndJunkFeatures

    X = np.array(carAndJunkFeatures.X)
    y = np.array(carAndJunkFeatures.y)

    testSize = .3
    randomState=1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    clf0 = svm.SVC(kernel='linear')
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])
    
    clf0.fit(X_train,y_train)
    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    clf3.fit(X_train,y_train)
    eclf.fit(X_train,y_train)
    
    estimatorName=['linear SVC', 'Decision Tree', 'K Neighbors', 'rbf Kernel SVC', 'Voting Classifier']
    predictions=[]
    predictions.append(clf0.predict(X_test))
    predictions.append(clf1.predict(X_test))
    predictions.append(clf2.predict(X_test))
    predictions.append(clf3.predict(X_test))
    predictions.append(eclf.predict(X_test))
    
    for i in range(5):

        pprint.pprint(estimatorName[i])
        pprint.pprint('Predictions:'), pprint.pprint(np.array(predictions[i]))
        pprint.pprint('Ground Truth:'), pprint.pprint(np.array(y_test))
        
        predVsTruth=predictions[i]==y_test        
        pprint.pprint(predVsTruth)
        numCases =(len(predictions[i]))
        numTrue = np.sum(predVsTruth)
        numFalse = numCases - numTrue
        print('Accuracy is: "%s"' % (numTrue/numCases*100))
        print('Number True: "%s", Number False: "%s"\n\n' % (numTrue,numFalse))

    #Must download Graphviz.exe and pip install graphviz for this to work
    #Gives a tree representation of the decision tree decision parameters. 
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
    dot_data = tree.export_graphviz(clf1, out_file=None) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("carClassifier.pdf") 

    #The graphing portion of this is not working 100% currently
    feat_labels = ['m','sf','mx','mi','sdev','amin','smin','stmin','apeak','speak','stpeak','acep','scep','stcep','aacep','sscep','stsscep','zcc','zccn','spread','skewness','savss','mavss']

    forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                feat_labels[indices[f]], 
                                importances[indices[f]]))

    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            color='lightblue', 
            align='center')
    for f in range(X_train.shape[1]):
        plt.xticks(range(X_train.shape[1]), 
                   feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('./random_forest.png', dpi=300)
    plt.show()
    
def getRms(frames):
    '''
    Gets the normalized RMS value from a numpy array of audio data
    
    Parameters
    ----------
    frames : numpy array of audio data

    Returns
    -------
    rms : the normalized average RMS value over the length of the frames
    '''
    frames = frames / (2.0 ** 15)  # convert it to floating point between -1 and 1
    rms = np.sqrt(np.mean(np.square(frames)))
    return rms
    
def calc_slope(x,y):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

def get_indicators(vec):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    std = np.std(vec)
    return mean, slope, std

def calc_u(vec):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    fft = np.fft.fft(vec)
    return np.sum(np.multiply(fft,vec))/np.sum(vec)

def calc_features(vec,freq):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    #bin count
    bc = 10
    bincount = list(range(bc))
    #framesize
    fsize = 512
    #mean
    m = np.mean(vec)
    #spectral flux
    sf = np.mean(vec-np.roll(vec,fsize))
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    binwidth = int(len(vec)/bc)
    bins = []
    for i in range(0,bc):
        bins.append(vec[(i*binwidth):(binwidth*i + binwidth)])
    peaks = [np.max(i) for i in bins]
    mins = [np.min(i) for i in bins]
    amin,smin,stmin = get_indicators(mins)
    apeak, speak, stpeak = get_indicators(peaks)
    #fft = np.fft.fft(vec)
    bin_fft = []
    for i in range(0,bc):
        bin_fft.append(np.fft.fft(vec[(i*binwidth):(binwidth*i + binwidth)]))

    cepstrums = [np.fft.ifft(np.log(np.abs(i))) for i in bin_fft]
    inter = [get_indicators(i) for i in cepstrums]
    acep,scep, stcep = get_indicators([i[0] for i in inter])
    aacep,sscep, stsscep = get_indicators([i[1] for i in inter])

    zero_crossings = np.where(np.diff(np.sign(vec)))[0]
    zcc = len(zero_crossings)
    zccn = zcc/freq

    u = [calc_u(i) for i in bins]
    spread = np.sqrt(u[-1] - u[0]**2)
    skewness = (u[0]**3 - 3*u[0]*u[5] + u[-1])/spread**3

    #Spectral slope
    #ss = calc_slope(np.arange(len(fft)),fft)
    avss = [calc_slope(np.arange(len(i)),i) for i in bin_fft]
    savss = calc_slope(bincount,avss)
    mavss = np.mean(avss)

    return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness,savss,mavss]

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min = np.real(x1_min)
    x1_max = np.real(x1_max)
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x2_min = np.real(x2_min)
    x2_max = np.real(x2_max)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

if __name__=='__main__':
    #renameFiles()
    #getRandTestSet()
    #determineFeatures('car')
    #determineFeatures('junk')
    trainClassifier()
