function [U]=function_Unsupervised_Learning_LR(X,k,k_keep,iter)
[m,n]=size(X);
H_k=(-1/(k+1)).*ones(k+1,k+1)+diag(ones(1,k+1));%H_k+1
F=function_knn_cal(X,k);
F_=zeros(n,1);
for i=1:n
    F_(i)=i;
end
F=[F_,F];
M=zeros(m,m);
for i=1:n
    X_i=[X(:,i)];
    for j=2:k+1
        X_i=[X_i,X(:,F(i,j))];
    end
    B_i=pinv((X_i*H_k)'*(X_i*H_k)+eye(k+1)*10^(-5));
    S_i=zeros(n,k+1);
    for p=1:k+1
        S_i(F(i,p),p)=1;
    end
    M_i=S_i*H_k*B_i*H_k*S_i';
    M=M+X*M_i*X';
end
D_t=diag(ones(1,m));
U=diag(ones(1,m));
for t=1:iter
    P_t=M+D_t;
    [new_U,d]=eig(P_t);
    [~,order]=sort(diag(d),'ascend');
    new_U=new_U(:,order);
    D_t=zeros(m,m);
    for i=1:m
        D_t(i,i)=1/(2*norm(new_U(i,1:k_keep),2));
    end
    new_U=new_U';
    delta_U=norm(U(1:k_keep,:)-new_U(1:k_keep,:),2);
    U=new_U;
    fprintf('iter=%d,delta_U=%f\n',t,delta_U);
    if(delta_U<1)
        fprintf('break\n');
        break;
    end
end
end