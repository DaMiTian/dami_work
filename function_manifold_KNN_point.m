function [K_point,K]=function_manifold_KNN_point(Data,point,k)
[m,n]=size(Data);
K_point=zeros(1,k);
dist=zeros(1,n);
for i=1:n
    for j=1:m
        dist(1,i)=dist(1,i)+(Data(j,i)-point(j,1))^2;
    end
end
[~,order]=sort(dist,'ascend');
K_point=zeros(m,k);
K=order(2:k+1);
for i=2:k+1
    K_point(:,i-1)=Data(:,order(i));
end
end