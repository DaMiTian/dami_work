function U=function_SRLDA(X,k_keep,cycle,ALL)
mean_X=X-mean(X,2);
[m,n]=size(X);
W=zeros(n,n);
w=ones(cycle,cycle)./cycle;
for i=1:ALL
    W((i-1)*cycle+1:i*cycle,(i-1)*cycle+1:i*cycle)=w;
end
W=W*W';
[y,D]=eig(W);
[~,order]=sort(diag(D),'descend');
y=y(:,order);
y=y(:,1:k_keep);
a=pinv(mean_X*mean_X'+diag(ones(m)))*mean_X*y;
U=a';
end
