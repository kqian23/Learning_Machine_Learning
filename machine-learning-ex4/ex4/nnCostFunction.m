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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------
% Part 1 - perform forward propagation and return the cost

X = [ones(size(X,1),1) X];

% Layer 1
a1 = X;

% Layer 2
z2 = a1 * Theta1'
a2 = sigmoid(z2)

% Layer 3
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);


% convert y to 0 and 1 encoding
new_y = zeros(size(y,1), num_labels);

for c = 1:size(y,1),
	digit = y(c);
	new_y(c,digit) = 1;
end;

% Compute cost
j_temp = 0;

J = (1/m) * sum ( sum (  (-new_y) .* log(a3)  -  (1-new_y) .* log(1-a3) ));


% Introduce Regularization
Theta1_reg = Theta1(:,2:size(Theta1,2));
Theta2_reg = Theta2(:,2:size(Theta2,2));


% Regularized Cost Function
r = 0
r = (lambda/(2*m))*(sum(sum(Theta1_reg.^2))+sum(sum(Theta2_reg.^2)))
J=J+r



% -------------------------------------------------------------
% Part 2 - Back Propagation

Delta = 0;
Delta_1 = 0;
Delta_2 = 0;

for t = 1:m,   % t corresponds to t th training example
	% Step 1
	a_1 = X(t,:); % Layer 1  1x401
	z_2 = a_1 * Theta1';  % Layer 2   1x25
	a_2 = sigmoid(z_2);    % 1x25
	a_2 = [ones(1, 1) a_2];   % Layer 3 1x26
	z_3 = a_2 * Theta2'; % 1x10
	a_3 = sigmoid(z_3); % 1x10

	% step 2 - compute delta for each of unit k in layer 3
	dlt_3 = a_3 - new_y(t,:);	% 1x10

	% step 3 - compute delta for layer 2
  z_2 = [1 z_2]
	dlt_2 = (Theta2' * dlt_3') .* sigmoidGradient(z_2)';  % 26x10, 10x1 -> 26x1

	% step 4 - accumulate gradient
	Delta_2 = Delta_2 + (dlt_3' * a_2); %  10x1, 1x25
	Delta_1 = Delta_1 + (dlt_2(2:end,:) * a_1); % 25x1, 1x401
end;

% step 5 - obtain unregularized gradient
Theta1_grad = (1/m) .* Delta_1; % 25x401
Theta2_grad = (1/m) .* Delta_2; % 10x25

% regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
