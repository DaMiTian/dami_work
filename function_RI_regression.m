function [U,iter_time]=function_RI_regression(trainData,W,Y,k_keep,iter)
iter_time=1;
flag=1;
Y=Y';
W=W(1:k_keep,:);
W=W';
E=Y-trainData'*W;
I=diag(ones(1,size(trainData,2)));
A=[trainData',I];
U=[W;E];
D=ones(1,size(A,2));
D=diag(D);
for t=0:iter
    U_t=pinv(D)*A'*pinv(A*pinv(D)*A')*Y;
    D=zeros(1,size(U_t,1));
    for i=1:size(U_t,1)
        u_=2*norm(U_t(i,:),2);
        if u_==0
            flag=0;
        end
        D(1,i)=1/u_;
    end
    D=diag(D);
    delta_U=norm(U-U_t,2);
    fprintf('iter:%d,delta_U=%f\n',t,delta_U);
    if delta_U<0.01 || flag==0
        iter_time=i;
        fprintf('end!\n');
        break;
    end
    U=U_t;
end
end