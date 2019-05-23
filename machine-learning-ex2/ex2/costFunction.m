function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % here we have the vector column zeros for gradient

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
## calculate the costFunction
for i=1:m % here i iterate the training set, we have m training set (x,y)
  J=J+ 1/m*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
endfor
## calculate the gradient

for j=1:size(theta)  % here i iterate the parameter element of theta, also means
  %  each element of vector row x of X 
  for i=1:m
    grad(j)= grad(j)+ 1/m*(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
  endfor
  % please notice that we have X(i,:)*theta because the first element is a row vector
  % and theta is a column vector
endfor







% =============================================================

end
