function [ J ] = least_square( X, y, theta )

    m = length(y);
    J = sum((X * theta - y) .^ 2)/ (2*m);
    
end
