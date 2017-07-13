function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


predictions = sigmoid(X * theta); %h predictions matrix  100x1  (100x3 x 3x1) [h0,h1,h2]
J = 1/m * sum((-y).*log(predictions)-((1-y).*(log(1-predictions)))); % cost real number elementwise * 100x1 * 100x1

grad = 1/m * X'*(predictions.-y); %gradients X' = 3x100  (each row = a feature) * 100x1 = 3x1  matrix * = sum of mat1_row(i)*mat2_col(i)


% =============================================================

end
