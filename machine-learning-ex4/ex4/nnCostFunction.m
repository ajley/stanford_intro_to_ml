function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m,1),X]; %adding bias = A1
A2 = sigmoid(X*Theta1'); % a2 = g(z2) z2 = theta1 * a1 X = mxn theta1 = s(1)xn 
A2 = [ones(m,1),A2]; %add bias
A3 = sigmoid(A2 * Theta2');  %this is h_theta:  g(z3) = theta2*a2 Ae = mxs(2)x s(3)x(s(2)
% so A3 in this example = mxs(3) = 5000x10

%y = 5000x1, need to change is to 5000x10 for the cost function.  needs to have 0s and 1 instead of 0-9

yy = zeros(m,num_labels);
for i = 1:m,
  yy(i,y(i)) = 1;
end
singlecostfunc = -yy.*log(A3) - (1-yy).*log(1-A3);


% zero out theta(0) in both theta levels
t1 =[zeros(size(Theta1,1),1),Theta1(:,2:size(Theta1,2))];
t2 =[zeros(size(Theta2,1),1),Theta2(:,2:size(Theta2,2))];
J = 1/m * sum(sum(singlecostfunc));
%set up regularization parameter  sum of squares of thetas (not bias)
regularizationParam = (lambda/(2*m))* (sum(sum(t1.^2)) + sum(sum(t2.^2)));

J = J + regularizationParam;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%backprog implementation below

for k =1:m,
  
  %Forward Propagation to L = 3
  x = X(k,:); % single test set of features with bias added  1xn+1 matrix
  z2 = (Theta1 * x');  %a2 = 25x1
  a2 = sigmoid(z2); %26x1
  a2 = [1 ; a2];
  z3 = (Theta2 * a2); %10x1
  a3 = sigmoid(z3);  %this is the hypothesis ( 10 x 1 )
    
  %Back Propagation
  d3 = a3 .- yy(k,:)';
  z2 = [1 ; z2];
  d2 = (Theta2' * d3) .* sigmoidGradient(z2);
  d2 = d2(2:end);
  %whos
  Theta2_grad = (Theta2_grad + d3 * a2');
  Theta1_grad = (Theta1_grad + d2 * x);
endfor


Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;  

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% adding in regularization term to Thetas

Theta1_grad = Theta1_grad + (lambda/m)* t1;

Theta2_grad = Theta2_grad + (lambda/m)* t2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
