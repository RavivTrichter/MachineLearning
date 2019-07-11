clear ; close all; clc

load('face_train.mat');
% vecToImage(Xtrain(1,:));
rows_Xtrain = size(Xtrain,1);
cols_Xtrain = size(Xtrain,2);
outputLayer = 2; % face or not

alphas = [2 , 5 , 7];
lambdas = [0.01, 0.03, 0.05];
hidden_layers = [10, 30, 15];
[X_train ,X_test, y_train, y_test] = CrossValidation(Xtrain, ytrain);
% [best_alpha, best_hidden_layer, best_lambda] = getParameters(X_train, y_train,outputLayer, alphas, lambdas, hidden_layers);
best_alpha = 5;
best_hidden_layer = 15; 
best_lambda = 0.03;
max_iter = 2500;
Theta1 = InitializeParam(cols_Xtrain, best_hidden_layer);
Theta2 = InitializeParam(best_hidden_layer, outputLayer);
fprintf('------------Finished Training - Running on Test Vectors------------\n');
% X_test and y_test are 30% of the training set
[J,Theta1,Theta2] = bp(Theta1, Theta2, X_test,y_test,max_iter, best_alpha, best_lambda);
load('face_test.mat');
fprintf('------ Loaded face_test.mat and now running it : -------\n');
ytest(ytest == -1) = 2;
ff_predict(Theta1,Theta2,Xtest,ytest);