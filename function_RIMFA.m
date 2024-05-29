%RIMFA

function [U]=function_RIMFA(allData,all_label,U_init,k,k_remain,time)
U=U_init;
K=function_manifold_KNN(allData,k);

[m,n]=size(allData);
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

X_R=zeros(m,n^2);
ptr=1;
for i=1:n
    for j=1:n
        X_R(:,ptr)=allData(:,i)-allData(:,j);
        ptr=ptr+1;
    end
end

G_c=zeros(1,n^2);
ptr=1;
for i=1:n
    for j=1:n
        G_c(ptr)=W_c(i,j);
        ptr=ptr+1;
    end
end
G_c=diag(G_c);

G_p=zeros(1,n^2);
ptr=1;
for i=1:n
    for j=1:n
        G_p(ptr)=W_p(i,j);
        ptr=ptr+1;
    end
end
G_p=diag(G_p);

for iter=1:time
    fprintf('iter:%d\n',iter);
    D_c=zeros(n^2,n^2);
    D_p=zeros(n^2,n^2);
    ptr=1;
    for i=1:n
        for j=1:n
            w=W_c(i,j)*sqrt((U(1:k_remain,:)*(X_R(:,ptr)))'*(U(1:k_remain,:)*(X_R(:,ptr))));
            if w~=0
            D_c(ptr,ptr)=1/w;
            end
            ptr=ptr+1;
        end
    end

    ptr=1;
    for i=1:n
        for j=1:n
            w=W_p(i,j)*sqrt((U(1:k_remain,:)*(X_R(:,ptr)))'*(U(1:k_remain,:)*(X_R(:,ptr))));
            if w~=0
            D_p(ptr,ptr)=1/w;
            end
            ptr=ptr+1;
        end
    end
    
    S_w=X_R*G_p*D_p*G_p*X_R';
    S_b=X_R*G_c*D_c*G_c*X_R';
   
    
    [V,D]=eig(pinv(S_w)*S_b);
    [~,order]=sort(diag(D),'ascend');
    V=V(:,order);
    V=V';
    
    delta_U=norm(U-V,2);
    fprintf("iter:%d,delta_U=%d\n",iter,delta_U);
    if delta_U<1
        break;
    end
    U=V;
end
end