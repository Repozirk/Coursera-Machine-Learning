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

% PART_1-----------------------------------------------------------------------

% CS: creates a matrix: "1" in colum1, "2" in colum2 ...
y_matrix = eye(num_labels)(y,:); 

% CS: blank variables to fill
X = [ones(m, 1) X];
h=zeros(size(X, 1), num_labels);
Jtemp=zeros(size(y_matrix,2), 1);
T1 = size(Theta1, 1);
T2 = size(Theta2, 1);


% CS: computing the output layer for every training set
for i=1:m;
  
      z2=(X(i,:)*Theta1');
      a2(i,:)=(1./(1.+e.^(-z2))); 
    
      a02 = [ones(i, 1), a2];
      
      z3=(a02(i,:)*Theta2');
      a3(i,:)=(1./(1.+e.^(-z3)));
      h=a3;
    
end


%size(a02)
%size(a3)
%size(h)

% CS: computing the Cost of the NN without regularization
for q=1:size(y_matrix,2);
  
  Jtemp(q,1)= sum(-y_matrix(:,q).*log(h(:,q)).-(1.-y_matrix(:,q)).*log(1.-h(:,q)));

end

J=1/m*(sum(Jtemp));

% CS: computing the Cost of the NN with regularization
%for q=1:size(y_matrix,2);
%  
%  Jtemp(q,1)= sum(-y_matrix(:,q).*log(h(:,q)).-(1.-y_matrix(:,q)).*log(1.-h(:,q)));
%  
%end

Theta1reg=Theta1;
Theta1reg(:,1)=0;
Theta2reg=Theta2;
Theta2reg(:,1)=0;

Regtemp1 = sum(sum(Theta1reg.^2));
%size (Regtemp1)
Regtemp2 = sum(sum(Theta2reg.^2));
%size (Regtemp2)
Regtemp = Regtemp1 .+ Regtemp2;

J=1/m*(sum(Jtemp)) + (lambda/(2*m)*Regtemp);


% -------------------------------------------------------------

% PART_2-----------------------------------------------------------------------


Delta1=0;
Delta2=0;
%a2=zeros(size(X,1),T1);
%a3=zeros(size(X,1),T2);

for t = 1:m;
      
      z2=(X(t,:)*Theta1');
      a2(t,:)=(1./(1.+e.^(-z2))); 
    
      a02 = [ones(m, 1), a2];
      
      z3=(a02(t,:)*Theta2');
      a3(t,:)=(1./(1.+e.^(-z3)));
      h=a3;
      
      d3(t,:)=(h(t,:).-y_matrix(t,:));
%      size(d3)
      r= size(Theta2,2);
      d2(t,:)=(d3(t,:)*Theta2(:,2:r)).*sigmoidGradient(z2); 
 
  
  Delta1=Delta1 .+ d2(t,:)'*X(t,:);
  Delta2=Delta2 .+ d3(t,:)'*a02(t,:);
  
end  

Theta1(:,1)=0;
Theta2(:,1)=0;
  Theta1_grad=1/m.*(Delta1).+(lambda/m*(Theta1));
  Theta2_grad=1/m.*(Delta2).+(lambda/m*(Theta2));
  

   
% size(a02)
% size(d3)
% size(d2)
% size(Delta1)
% size(Delta2)
% size(Theta1_grad)
% size(Theta2_grad)


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
 
end