function K=kernel(X,Y)
global sigma;
K = exp(-norm(X-Y)^2/(2*sigma));
end
