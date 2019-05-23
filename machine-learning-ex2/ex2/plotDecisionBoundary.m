function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y); % it means we get the column 2 and 3 of matrix X
hold on

if size(X, 2) <= 3 % it means we get the size of X at the dimension 2 ~ number of column 
% it means that the vector x is only composited of 2 elements, then the decision boundary
% is a linear line so we can plot it through equation draw
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2]; % just choose 2 points not important because
    % we have the equation of line below which determine in fact the decision boundary line

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
% here we have to understand that y is the x subscript 2= the element of third column
% in the X matrix . As we know , the boundary decision line means z=0, it equals to 
% x_0* theta(1) + x_1*theta(2)+ x_2*theta(3)=0 
% here we have x_0=1, x_1=plot_x, x_2= plot_y
% please notice that we count from 0 so we have in theory x_0,x_1,x_2 
% but the naming indexes are flexible and not important

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y) % it means plot data 
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else 
    % here, the decision line is more complex because it's the case more than variable x
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
