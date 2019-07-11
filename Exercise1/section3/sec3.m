load ('houses.txt')
[norm_X, avg, dev] = normalize(houses(:, 1 : 2));
norm_X = [ones(length(houses), 1) norm_X];
y = houses(:, end);
theta = [0;0;0];
max_iter = 2000;
alpha = 1e-2;
[theta, J] = gradient_decent(norm_X, y, theta, alpha, max_iter, @least_square, false);

% 3.b
sf = 1800;
rooms = 5;
result = ([sf rooms] - avg) ./ dev;
fprintf('3.b. : %f\n', theta(1) + theta(2) * result(1) + theta(3) * result(2)); 


% 3.c
X2 = [ones(length(houses), 1) houses(:, 1 : 2)];
theta_ = (X2' * X2)^(-1) * X2' * y;
fprintf('3.c. : %f\n', theta_(1) + theta_(2) * sf + theta_(3) * rooms); 
