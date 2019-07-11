function [ theta, J ] = gradient_decent( X, y, theta, alpha, max_iter, cost_function, plot_f)
    m = length(X); 
    J = zeros(max_iter, 1);
    
    for iter = 1 : max_iter
         
        % Keeping track of the cost function
        J(iter) = cost_function(X, y, theta);
        % Updating theta
        theta = theta - alpha * (1 / m) * ((X * theta - y)' * X)';
        if plot_f == true
            figure(2)
            hold on
            plot(theta(1), theta(2), 'gx', 'MarkerSize', 5, 'LineWidth', 2)
        end

    end    
    
end