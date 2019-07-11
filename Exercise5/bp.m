function [J,Theta1,Theta2, max_detectp] = bp(Theta1, Theta2, X, y, max_iter, alpha , Lambda)
% BP BackPropagation for training a neural network
% Input arguments
% Theta1 - matrix of parameters (weights)  between the input and the first hidden layer
% Theta2 - matrix of parameters (weights)  between the hidden layer and the output layer (or
% another hidden layer)
% X - input matrix
% y - labels of the input examples
% max_iter - maximum number of iterations (epochs).
% alpha - learning coefficient.
% Lambda - regularization coefficient.
%
% Output arguments
% J - the cost function
% Theta1 - updated weight matrix between the input and the first hidden layer
% Theta2 - updated weight matrix betwwn the hidden layer and the output layer (or
% second hidden layer)
%
% Usage:[J,Theta1,Theta2] = bp(Theta1, Theta2, X,y,max_iter, alpha,Lambda)
%

% Initalization
if nargin<7, Lambda=0; end
if nargin<6, alpha=0.1; end
if nargin<5, max_iter=1000; end
m = size(X, 1);
num_outputs = size(Theta2, 1);
delta3=zeros(size(Theta2,1),1); % delta's for the output layer
ybin=zeros(size(delta3));       % delta's for the hidden layer
delta2=zeros(size(Theta1,1),1);
p = zeros(size(X, 1), 1);
J=0;
max_detectp = 0;

for q=1:max_iter
    J=0;
    dTheta2=zeros(size(Theta2));
    dTheta1=zeros(size(Theta1));
    Theta2_grad=zeros(size(dTheta2));
    Theta1_grad=zeros(size(dTheta1));
    r=randperm(size(X,1));
    for k=1:m
        X1=X(r(k),:);
        % Forward propagation
        % -------------------
        X1=[ones(size(X1,1),1) X1];
        z2=X1*Theta1';
        a2=sigmoid(z2);
        a2=[ones(size(a2,1),1) a2];
        z3=a2*Theta2';
        a3=sigmoid(z3);
        % Backward propagation
        % ---------------------
        ybin=zeros(size(a3));
        if y(r(k)) < 0 , y(r(k))=2; end
        ybin(y(r(k)))=1; % Assigning 1 to the binary digit according to the class (digit). ybin(2)=1 for '0'.
        J=J+1/m*(-ybin*log(a3')-(1-ybin)*log(1-a3')); % Maximum likelihood cost function
        %J=J+(a3-ybin)*(a3-ybin)'; % Sum of least square cost function
        delta3=(a3-ybin)'; %.*(a3'.*(1-a3'));
        delta2=(Theta2'*delta3).*(a2'.*(1-a2'));
        % delta(2:end) because there is no Theta1(0,j)
        dTheta2=dTheta2+delta3*a2;
        dTheta1=dTheta1+delta2(2:end)*X1;% delta(2:end) because there is
                                         % no Theta1(0,j)
        
    end
    
    % updating J with the regularization cost
    J=J+(Lambda/(2*m))*(sum(sum((Theta1(:,2:end)).^2))+sum(sum((Theta2(:,2:end)).^2)));
    if mod(q,100)==0
        fprintf('\nCost function J= %f in iteration %d with Lambda = %d \n',J,q,Lambda);
        pause(0.0005)
    end
    Theta2_grad=1/m*dTheta2;
    Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+Lambda/m*Theta2(:,2:end);
    Theta1_grad=1/m*dTheta1;
    Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+Lambda/m*Theta1(:,2:end);
    Theta1=Theta1-alpha*Theta1_grad; % Updating the parameters (weights)
    Theta2=Theta2-alpha*Theta2_grad;
    
    if mod(q,50)==0
        [p, detectp] = ff_predict(Theta1, Theta2, X,y);
        fprintf('\nNetwork Accuracy : %f \n', detectp);
        if detectp > max_detectp
            max_detectp = detectp;
        end
        fprintf('alpha : %f, Lambda : %f, Hidden Layers : %d \n', alpha, Lambda, size(Theta2,2));
    end
   
end











