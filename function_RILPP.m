%RILPP

function U=function_RILPP(allData,U_init,k,k_remain,time)
U=U_init;
K=function_manifold_KNN(allData,k);
[m,n]=size(allData);

W=zeros(n,n);
for i=1:n
    for j=1:k
        W(i,K(i,j))=1;
        W(K(i,j),i)=1;
    end
end
mean_length=mean(allData,2)'*mean(allData,2);
mean_length=mean_length/10;

for i=1:n
    for j=1:n
        if W(i,j)==1
            W(i,j)=exp(-(allData(:,i)-allData(:,j))'*(allData(:,i)-allData(:,j))/mean_length);
        end
    end
end

D=diag(sum(W));

G=zeros(1,n^2);
ptr=1;
for i=1:n
    for j=1:n
        G(ptr)=W(i,j);
        ptr=ptr+1;
    end
end
G=diag(G);

X_R=zeros(m,n^2);
ptr=1;
for i=1:n
    for j=1:n
        X_R(:,ptr)=allData(:,i)-allData(:,j);
        ptr=ptr+1;
    end
end

for iter=1:time
    fprintf('iter:%d\n',iter);
    D_Rl=zeros(1,n^2);
    ptr=1;
    for i=1:n
        for j=1:n
            w=W(i,j)*sqrt((U(1:k_remain,:)*(X_R(:,ptr)))'*(U(1:k_remain,:)*(X_R(:,ptr))));
            if w~=0
            D_Rl(1,ptr)=1/w;
            end
            ptr=ptr+1;
        end
    end
    D_Rl=diag(D_Rl);
    
    D_Rd=zeros(1,n);
    for i=1:n
        w=D(i,i)*sqrt((U(1:k_remain,:)*(allData(:,i)))'*(U(1:k_remain,:)*(allData(:,i))));
        if w~=0
            D_Rd(i)=1/(w);
        end
    end
    D_Rd=diag(D_Rd);
    
    S_Rl=X_R*G*D_Rl*G*X_R';
    S_Rd=allData*D_Rd*allData';
    
    [V,d]=eig(pinv(S_Rd)*S_Rl);
    [~,order]=sort(diag(d),'ascend');
    V=V(:,order);
    V=V';
    
    delta_U=norm(U-V,2);
    fprintf("iter:%d,delta_U=%d\n",iter,delta_U);
    
    if iter>5 && delta_U<1
        break;
    end
    U=V;
end
end