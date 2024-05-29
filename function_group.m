function [group]=function_group(X,cycle,i)
group=X(:,(i-1)*cycle+1:i*cycle);
end