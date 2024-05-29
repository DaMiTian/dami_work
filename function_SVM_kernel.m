function [alpha,b]=function_SVM_kernel(X,Y)
% 训练核SVM模型
h=zeros(size(X,2),size(X,2));
for i=1:size(X,2)
    for j=1:size(X,2)
        h(i,j)=kernel(X(:,i),X(:,j))*Y(i)*Y(j);
    end
end
f=-1*ones(size(X,2),1);%目标函数的f

%等式约束 aeq*x=beq
aeq=Y;
beq=zeros(1,1);
%自变量约束lb(i)<=x(i)<=ub
lb=zeros(size(X,2),1);
alpha = quadprog(h,f,[],[],aeq,beq,lb,[]); % 使用二次规划求解alpha值
%b = mean(Y - K(X,X) * alpha); % 计算偏置项b
b=0;
for i=1:size(X,2)
    for j=1:size(X,2)
    b=b+Y(i)-kernel(X(:,i),X(:,j))*alpha;
    end
end
b=b/(size(X,2)^2);
end