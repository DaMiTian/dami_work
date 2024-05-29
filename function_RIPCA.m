%RIPCA

function [U]=function_RIPCA(data,U_init,k_remain,iter)
[~,n]=size(data);
U=U_init;%初始化特征向量矩阵

for i=1:iter
    fprintf('iter:%d\n',i);
    D=zeros(n,n);
    meanvalue=ones(size(data,1),1)*mean(data);
    for j=1:n
        D(j,j)=1/sqrt(((data(:,j)-meanvalue(:,j))'*U(1:k_remain,:)')*((data(:,j)-meanvalue(:,j))'*U(1:k_remain,:)')');
    end
    [newU,V]=eig(data*D*data');
    v=diag(V);
    [~,order]=sort(v,"descend");
    newU=newU(:,order);
    delta_U=norm(U-newU',2);
    fprintf("iter:%d,delta_U=%d\n",i,delta_U);
    if delta_U<1
        fprintf('break,%d\n',i);
        break;
    end
    U=newU';
end
end