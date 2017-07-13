function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

yeses = find(y==1);
noes = find(y==0);

plot(X(yeses, 1), X(yeses, 2), 'kd', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

plot(X(noes, 1), X(noes, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);






% =========================================================================



hold off;

end
