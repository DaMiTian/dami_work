
function K=function_manifold_KNN(allData,k)
[n,m]=size(allData);
K=zeros(m,k);
for p=1:m
    dist=zeros(m,1);
    for j=1:m
        Dist=0;
        for i=1:n
            Dist=Dist+abs(allData(i,p)-allData(i,j));
        end
        dist(j,1)=Dist;
    end
    [~,A]=sort(dist);
    B=[];
    for i=2:k+1
        B=[B,A(i,1)];
    end
    K(p,:)=B;
end
end