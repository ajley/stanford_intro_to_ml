function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

naturale = e.*ones(size(z,1),size(z,2));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

naturale = e.*ones(size(z,1),size(z,2));

g = 1./(1+exp(-z));


% =============================================================

end
