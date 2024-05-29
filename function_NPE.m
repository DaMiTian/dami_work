function [U]=function_NPE(X,k)
[~,n]=size(X);
K=function_manifold_KNN(X,k);
W=zeros(n,n);
one_k=ones(k,1);
for i=1:n
    Zi=[];
    for j=1:k
        Zi=[Zi,X(:,i)-X(:,K(i,j))];%Zi m*k
    end
    Zi=Zi'*Zi;%Zi k*k
    Zi=pinv(Zi);
    Wi=(Zi*one_k)/(one_k'*Zi*one_k);%Wi k*1
    for j=1:k
        W(i,K(i,j))=Wi(j,1);
    end
end
W=W';
M=(diag(ones(1,n))-W)*(diag(ones(1,n))-W)';
[V,D]=eig(pinv(X*X')*X*M*X');
[~,order]=sort(diag(D),'ascend');
V=V(:,order);
U=V';
end