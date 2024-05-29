function [U]=function_LR_improve(Y,X)
m1=size(Y,1);
m2=size(X,1);
n=size(Y,2);
W=diag(randn(1,n));
tmp=10000;
U=zeros(m1,m2);
iter=0;
while tmp>0.1
    new_U=Y*W*X'*pinv(diag(ones(1,m2))+X*W*X');
    max=0;
    for i=1:n
        W(i,i)=norm(Y(:,i)-new_U*X(:,i),2);
        if W(i,i)>max
            max=W(i,i);
        end
    end
    W=W./max;
    for i=1:n
        W(i,i)=exp(W(i,i));
    end
    tmp=tmp*0.9;
    delta_U=norm(U-new_U,2);
    fprintf('delta_U:%f,iter:%d\n',delta_U,iter);
    U=new_U;
    iter=iter+1;
    if delta_U<1
        fprintf('break!\n');
        break;
    end
end
end