# Movie-Genre-Prediction

link: https://huggingface.co/spaces/competitions/movie-genre-prediction
This repository present the solution which allow to get 1st rank to this competition.

## dataset
The goal of this competition is to design a predictive model that accurately classifies movies into their respective genres based on their titles and synopses.
The dataset contains a mix of both original and AI-generated titles, genres, and synopses to test the robustness of the models.
The 10 genres include action, adventure, crime, family, fantasy, horror, mystery, romance, scifi, and thriller.

## Solution:

The solution is based on:

- a domain transfer on roberta-large using Mask Language modeling.
- a roberta-large finetuned using 15-folds
- a roberta large finetuned using 15-folds using pseudo label on Test set (improve only public LB)
- use of TeacherFreeLoss function which simulate a virtual teacher
- Average pooling instead of using CLS token

0) download data: Create_folds.ipynb

1) apply a domain transfer on roberta-large : MLM.ipynb
The checkpoint of the MLM of roberta-large on movie dataset can be found here : https://huggingface.co/Shiro/roberta-large-movie-genre

2) finetune roberta-large 15-fold and generate prediction on test set : Classifier-roberta-ft-15fold-tf.ipynb

3) finetune roberta-large 15 folds using the pseudo label on the test set, semi-supervised learning approach : Classifier-roberta-ft-15fold-tf-PL.ipynb

## Results

|Model 15 fold| CV | Public LB  | Private LB  | 
|---|---|---|---| 
|roberta-large + MLM   | 0.445  | 0.4456  | 0.4412  | 
|roberta-large + MLM + PL   | 0.456  | 0.4456  | 0.4412  | 


## package

transformers=4.21.3

torch=1.12