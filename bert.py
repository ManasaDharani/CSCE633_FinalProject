import numpy as np 
import os

from pytorch_pretrained_bert.modeling import BertForSequenceClassification,BertConfig
from pytorch_pretrained_bert import BertTokenizer
from fastai.text import * 
from fastai.callbacks import CSVLogger
from fastai.callbacks import *

import pandas as pd 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support as score

#Read Training and testing Data
training_data=pd.read_csv('../input/train.csv');
testing_data=pd.read_csv('../input/test.csv');
punctuation_list=[];
import string
for traindata in training_data['question_text']:
	for i in range(0,len(traindata)):
		if(traindata[i] in string.punctuation):
			traindata.replace(traindata[i],' ');
		if(traindata[i].isdigit()):
			traindata.replace(traindata[i],'#');
	punctuation_list.append(traindata);
training_data['question_text']=punctuation_list;

#Split Training and Validation
training_data.rename(columns={'target':'label', 'question_text':'text'},inplace=True)
temp=training_data[['label','text']]
temp['1']=temp['label'].apply(lambda x: 1 if x==1 else 0);
temp['0']=temp['label'].apply(lambda x: 1 if x==0 else 0);
training_data=temp[:int(len(temp)*.80)]
validation_data=temp[int(len(temp)*.80):]


tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
#Wrapper around BERT to make it compatible with Fastai
class wrapper_bert_fastai(BaseTokenizer): 
    def __init__(self,tokenizer:BertTokenizer,len_max:int=128,**kwargs): 
         self.bert_tok=tokenizer 
         self.len_max=len_max
    def __call__(self,*args,**kwargs): 
         return self 
    def tokenizer(self,t:str)->List[str]: 
         return ["[CLS]"]+self.bert_tok.tokenize(t)[:self.len_max - 2]+["[SEP]"]

#Fastai tokenizer and vocabulary definitions       
token_bert_fastai=Tokenizer(tok_func=wrapper_bert_fastai(tokenizer,max_seq_len=256),pre_rules=[],post_rules=[])
vocab_bert_fastai=Vocab(list(tokenizer.vocab.keys()))
training_data.rename(columns={'target':'label','question_text':'text'},inplace=True)

label_cols=["1","0"] 
#Creating a textatabunch
textdatabunch=TextDataBunch.from_df(".",training_data,
                                  validation_data,
                                  validation_data,
                                  include_bos=False,
                                  include_eos=False, 
                                  tokenizer=token_bert_fastai,
                                  vocab=vocab_bert_fastai,
                                  text_cols="text",
                                  bs=32,
                                  label_cols=label_cols,
                                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),)

#Define Model and Loss
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
loss=nn.BCEWithLogitsLoss()
model_learner=Learner(textdatabunch,model,callback_fns=[partial(CSVLogger,append=True)],loss_func=loss)

#Train the model
model_learner.fit_one_cycle(1,3e-5)
#MULTICLASS
#Get Predicted Labels
prediction=model_learner.get_preds(DatasetType.Valid)[0].detach().cpu().numpy()
y_prediction=model_learner.get_preds(DatasetType.Valid)[1].detach().cpu().numpy()
prediction=model_learner.get_preds(DatasetType.Valid)[0].detach().cpu().numpy()
prediction_y=model_learner.get_preds(DatasetType.Valid)[1].detach().cpu().numpy()   
temp=[]
for s in textdatabunch.dl(DatasetType.Valid).sampler:
    temp.append(s);
temp_rev=np.argsort(temp);
prediction=prediction[temp_rev, :];
estimated_prediction=prediction_y[temp_rev, :]
max_index=np.argmax(prediction,axis=-1);
#Extract Correct Predictions
correct_predictions=np.zeros(prediction.shape);
correct_predictions[np.arange(prediction.shape[0]),max_index]=1;
#Accuracy
accuracy=accuracy_score(correct_predictions,estimated_prediction);
max_index=np.argmax(correct_predictions,axis=-1);

#SINGLECLASS
#Get Correct Labels
single_label_correct_predictions=np.zeros((correct_predictions.shape[0],1));
for x in range(len(max_index)):
    if max_index[x]==0:
        single_label_correct_predictions[x]=1;
max_index=np.argmax(estimated_prediction,axis=-1);
#Get Predicted Labels
single_label_estimated_prediction=np.zeros((estimated_prediction.shape[0],1));

#Single Label Accuracy and F1 Score
for x in range(len(max_index)):
    if max_index[x]==0:
        single_label_estimated_prediction[x]=1;
single_label_accuracy=accuracy_score(single_label_correct_predictions,single_label_estimated_prediction);
print("Accuracy",single_label_accuracy);
f1Score=f1_score(single_label_correct_predictions,single_label_estimated_prediction);
print("F1Score",f1Score);