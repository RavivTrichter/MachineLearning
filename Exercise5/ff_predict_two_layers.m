function [ p, detectp ] = ff_predict_two_layers(Theta1, Theta2, Theta3, X, y)
% As ff_predict works for one layer only now it works for two hidden
% layers.
% Initializations
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    p = zeros(size(X, 1), 1);
    X1 = [ones(size(X,1),1) X];

    z2 = X1*Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1),1) a2];

    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    a3 = [ones(size(a3,1),1) a3];
    z4 = a3 * Theta3';
    a4 = sigmoid(z4);
    
    [k, p] = max(a4');
    p = p';
    detectp = (sum(p == y) / m * 100);
end

