function [X_train,X_test, y_train, y_test] = CrossValidation(X, y)
rows_X=size(X,1);
cols_X=size(X,2);
[train] = crossvalind('HoldOut',rows_X,0.3);
X_train = zeros(0.7*rows_X ,cols_X);
y_train = zeros(0.7*rows_X ,1);
X_test = zeros(0.3*rows_X ,cols_X);
y_test = zeros(0.3*rows_X ,1);
X_train = X(train == 1, :); % 3500 vectors
X_test = X(train == 0, :); % 1500 vectors
y_train = y(train == 1);
y_test = y(train == 0);




