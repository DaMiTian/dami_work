function [W]=function_LR(Y,X)
W=Y*X'*pinv(X*X');
end