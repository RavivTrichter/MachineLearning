function [best_alpha, best_hidden_layer, best_lambda] = getParameters(X, y, output_layer, alphas, lambdas, hidden_layers)
% X - the feature vector matrix
% y - the tags of each vector in X
% output_layer - the number of neurons in output layer
% alphas - a set of alpha values recommended
% lambdas - a set of lambda optional values
% hidden_layers - a set of options for hidden_layers

% ------ return values -------
% best_alpha - learning rate - the best of the options
% best_hidden_layers - the number of hidden layers
% best_lambda - Regularization paramter

% These parameters were recommended by the websites I added to the PDF

cols_X = size(X,2);
best_alpha = 0;
best_hidden_layer = 0;
best_lambda = 0;
max_iter = 100;
detect_p_max = 0;
for i=1:length(alphas)
    for j=1:length(lambdas)
        for k=1:length(hidden_layers)
            Theta1 = InitializeParam(cols_X, hidden_layers(k));
            Theta2 = InitializeParam(hidden_layers(k), output_layer);
            [J,Theta1_,Theta2_, detectp] = bp(Theta1, Theta2, X,y,max_iter, alphas(i),lambdas(j));
            if detectp > detect_p_max
                detect_p_max = detectp;
                best_alpha = alphas(i);
                best_lambda = lambdas(j);
                best_hidden_layer = hidden_layers(k);
            end
        end
    end
end