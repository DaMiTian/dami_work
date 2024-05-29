%MFA

function [U]=function_MFA(allData,all_label,k)

K=function_manifold_KNN(allData,k);

[~,n]=size(allData);
W_c=zeros(n,n);
W_p=zeros(n,n);

for i=1:n
    for j=1:k
        if all_label(K(i,j))==all_label(i)
            W_c(i,K(i,j))=1;
        else
            W_p(i,K(i,j))=1;
        end
    end
end

D_c=diag(sum(W_c));
D_p=diag(sum(W_p));

S_w=allData*(D_p-W_p)*allData';
S_b=allData*(D_c-W_c)*allData';

[U,D]=eig(pinv(S_w)*S_b);
[~,order]=sort(diag(D),'ascend');
U=U(:,order);
U=U';
end