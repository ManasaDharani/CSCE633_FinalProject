import numpy as np
import string
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from sklearn import ensemble,metrics,naive_bayes,model_selection,linear_model,preprocessing


training_data=pd.read_csv("train.csv");
testing_data=pd.read_csv("test.csv");


wordcountlist=[];
charcountlist=[];
specialcountlist=[];
wordlenavglist=[];
uppercaselist=[];
uniquecountlist=[];
titlecountlist=[];
import string
for traindata in training_data['question_text']:
    #print(traindata)
    charcount=0;
    numcount=0;
    specialchars=0;
    avglen=0;
    uppercount=0;
    titlewords=0;
    #wordlist=[];
    wordcountlist.append(len(traindata.split()));
    uniquecountlist.append((len(set(traindata.split()))));
    wordlist =traindata.split();
    #print(wordlist)
    for i in range(0,len(wordlist)):
        avglen=avglen+len(wordlist[i]);
        #print(wordlist[i])
        if(wordlist[i].istitle()):
            titlewords=titlewords+1;
    avglen=avglen/len(wordlist);
    avglen=int(avglen);
    wordlenavglist.append(avglen);
    titlecountlist.append(titlewords); 
    for  i in range(0,len(traindata)):
        if(traindata[i].isalpha()):
            charcount = charcount+1;
        if(traindata[i] in string.punctuation):
            specialchars=specialchars+1;
        if(traindata[i].isupper()):
            uppercount=uppercount+1;
    uppercaselist.append(uppercount);
    specialcountlist.append(specialchars);
    charcountlist.append(charcount)

training_data['wordCount']=wordcountlist;
training_data['charCount']=charcountlist;
training_data['specialChars']=specialcountlist;
training_data['wordLenAvg']=wordlenavglist;
training_data["capitalWords"]=uppercaselist;
training_data["wordCount2"]=uniquecountlist;
training_data["wordCount3"]=titlecountlist;

#print(training_data.head());

#tfidf vector
vector = TfidfVectorizer(stop_words='english',ngram_range=(1,5));
vector.fit_transform(training_data['question_text'].values.tolist()+testing_data['question_text'].values.tolist());
train_vector=vector.transform(training_data['question_text'].values.tolist());
test_vector=vector.transform(testing_data['question_text'].values.tolist());


labels=training_data["target"].values;


wordcountlist=[];
charcountlist=[];
specialcountlist=[];
wordlenavglist=[];
uppercaselist=[];
uniquecountlist=[];
titlecountlist=[];
import string
for traindata in testing_data['question_text']:
    #print(traindata)
    charcount=0;
    numcount=0;
    specialchars=0;
    avglen=0;
    uppercount=0;
    titlewords=0;
    wordlist=[];
    wordcountlist.append(len(traindata.split()));
    uniquecountlist.append((len(set(traindata.split()))));
    wordlist = traindata.split();
    for i in range(0,len(wordlist)):
        avglen=avglen+len(wordlist[i]);
        if(wordlist[i].istitle()):
            titlewords=titlewords+1;
    avglen=avglen/len(wordlist);
    avglen=int(avglen);
    wordlenavglist.append(avglen);
    titlecountlist.append(titlewords); 
    for  i in range(0,len(traindata)):
        if(traindata[i].isalpha()):
            charcount = charcount+1;
        if(traindata[i] in string.punctuation):
            specialchars=specialchars+1;
        if(traindata[i].isupper()):
            uppercount=uppercount+1;
    uppercaselist.append(uppercount);
    specialcountlist.append(specialchars);
    charcountlist.append(charcount)

testing_data['wordCount']=wordcountlist;
testing_data['charCount']=charcountlist;
testing_data['specialChars']=specialcountlist;
testing_data['wordLenAvg']=wordlenavglist;
testing_data["capitalWords"]=uppercaselist;
testing_data["wordCount2"]=uniquecountlist;
testing_data["wordCount3"]=titlecountlist;

#print(testing_data.head());
def trainModel(train_X, labels, test_X, test_y, test_X2):
    trainedModel=linear_model.LogisticRegression(C=5.,solver='sag');
    trainedModel.fit(train_X,labels);
    testrediction=trainedModel.predict_proba(test_X)[:,1];
    testrediction2=trainedModel.predict_proba(test_X2)[:,1];
    return testrediction,testrediction2,trainedModel;
kfoldScore=[];
finalPrediction=0;
trainPrediction=np.zeros([training_data.shape[0]]);
kfoldValidation=model_selection.KFold(n_splits=10,shuffle=True,random_state=0);
for ind1,ind2 in kfoldValidation.split(training_data):
	x1,x2=train_vector[ind1],train_vector[ind2];
	y1,y2=labels[ind1],labels[ind2];
	kfoldPrediction,testPrediction,trainedModel=trainModel(x1,y1,x2,y2,test_vector);
	#print(kfoldPrediction);
	finalPrediction=finalPrediction+testPrediction;
	trainPrediction[ind2]=kfoldPrediction;
	kfoldScore.append(metrics.log_loss(y2,kfoldPrediction));

maxval=0-float('inf');
maxthreshold=0-float('inf');
for threshold in np.arange(0.1,0.3,0.01):
    threshold=np.round(threshold,1);
    currentPrediction=metrics.f1_score(y2,(kfoldPrediction>threshold).astype(int));
    #print(currentPrediction)
    if(currentPrediction>maxval):
        maxval=currentPrediction;
        maxthreshold=threshold;
    
print("f1score",maxval);



