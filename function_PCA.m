%PCA

function [newV]=function_PCA(X)
[~,m]=size(X);
X=X-mean(X,2);
cov=X*X'/m;
[V,D]=eig(cov);
d=diag(D);
[~,order]=sort(d,'descend');
V=V(:,order);
newV=V';
end