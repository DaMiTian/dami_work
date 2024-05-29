%LPP

function V=function_LPP(allData,k)
K=function_manifold_KNN(allData,k);
W=zeros(size(allData,2),size(allData,2));
for i=1:size(allData,2)
    for j=1:k
        W(i,K(i,j))=1;
        W(K(i,j),i)=1;
    end
end
mean_length=mean(allData,2)'*mean(allData,2);
mean_length=mean_length/10;

for i=1:size(allData,2)
    for j=1:size(allData,2)
        if W(i,j)==1
            W(i,j)=exp(-(allData(:,i)-allData(:,j))'*(allData(:,i)-allData(:,j))/mean_length);
        end
    end
end

D=diag(sum(W));

L=D-W;

[V,d]=eig(pinv(allData*D*allData')*allData*L*allData');
[~,order]=sort(diag(d),'ascend');
V=V(:,order);
V=V';
end