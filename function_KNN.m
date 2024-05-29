%knn for accuracy

function [Right]=function_KNN(testData,trainData,test_label,train_label,K)
error=0;
[n,m]=size(testData);
[~,M]=size(trainData);

for k=1:m
    dist=zeros(M,1);
    for j=1:M
        Dist=0;
        for i=1:n
            Dist=Dist+(testData(i,k)-trainData(i,j))^2;
        end
        dist(j,1)=Dist;
    end

[~,A]=sort(dist);

for i=1:M
    A(i,1)=train_label(1,A(i,1));
end

B=[];
for i=1:K
    B=[B,A(i,1)];
end

[a,b]=mode(B);
if b==1
    a=B(1,1);
end

test_label(2,k)=a;
if test_label(2,k)~=test_label(1,k)
    error=error+1;
end
end

Right=(m-error)/m*100;

end
