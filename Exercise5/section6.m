clear ; close all; clc

load('face_train.mat');
% vecToImage(Xtrain(1,:));
rows_Xtrain = size(Xtrain,1);
cols_Xtrain = size(Xtrain,2);
outputLayer = 2; % face or not

alphas = [2 , 4];
lambda = 0.05;
hidden_layers = [4,8];
[X_train ,X_test, y_train, y_test] = CrossValidation(Xtrain, ytrain);
%[best_alpha, best_first_layer,best_second_layer, best_lambda] = getParametersTwoLayers(X_train, y_train,outputLayer, alphas, lambda, hidden_layers);
best_alpha = 4; 
best_first_layer = 4;
best_second_layer = 4;
best_lambda = 0.05;
max_iter = 3000;
Theta1 = InitializeParam(cols_Xtrain, best_first_layer);
Theta2 = InitializeParam(best_first_layer, best_second_layer);
Theta3 = InitializeParam(best_second_layer, outputLayer);
fprintf('------------Finished Training - Running on Test Vectors------------\n');
% X_test and y_test are 30% of the training set
[ J, Theta1, Theta2, Theta3, max_detectp] = bp_two_layers(Theta1, Theta2,Theta3,X_test,y_test,max_iter, best_alpha, best_lambda);
load('face_test.mat');
fprintf('------ Loaded face_test.mat and now running it : -------\n');
ytest(ytest == -1) = 2;
ff_predict_two_layers(Theta1,Theta2,Theta3,Xtest,ytest);