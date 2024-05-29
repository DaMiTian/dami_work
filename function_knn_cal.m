function K=function_knn_cal(X,k)%统计前k近邻的数据
[m,n]=size(X);
K=zeros(n,k);
for p=1:n
    dist=zeros(n,1);
    for j=1:n
        Dist=0;
        for i=1:m
            Dist=Dist+(X(i,p)-X(i,j))^2;
        end
        dist(j,1)=Dist;
    end
    %距离排序
    [~,A]=sort(dist);
    %存储前K个数据对应的序号
    B=[];
    for i=2:k+1
        B=[B,A(i,1)];
    end
    K(p,:)=B;
end
end