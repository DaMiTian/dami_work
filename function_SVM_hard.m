function [W,b]=function_SVM_hard(X,y)
% [~,n]=size(X);
% h=zeros(n,n);
% for i=1:n
%     for j=1:n
%         h(i,j)=X(:,i)'*X(:,j)*y(i)*y(j);
%     end
% end
% f=-1*ones(n,1);%目标函数的f
% 
% %等式约束 aeq*x=beq
% aeq=y;
% beq=zeros(1,1);
% %自变量约束lb(i)<=x(i)<=ub
% lb=zeros(n,1);
% [a,~]=quadprog(h,f,[],[],aeq,beq,lb,[]);%二次规划问题
% 
% % for i=1:length(a)
% %     if a(i)<1e-10
% %         a(i)=0;
% %     end
% % end
% 
% w=0;%系数矩阵
% u=0;
% % ff=find(a~=0);
% % j=ff(1);%寻找a系数不等于0的下标j
% [~,j]=max(a);
% for i=1:n%关键系数数求解
%     w=w+a(i)*y(i)*X(:,i);
%     u=u+a(i)*y(i)*(X(:,i)'*X(:,j));
% end
% b=y(j)-u;
% X: 数据矩阵，每行是一个样本
% y: 标签矩阵，1或-1
% W: 权重向量
% b: 偏置
X=X';
[n, d] = size(X);

% 构造优化目标函数
H = diag([ones(d, 1); 0]);
f = zeros(d+1, 1);
A = -diag(y) * [X, ones(n, 1)];
b = -ones(n, 1);
lb = [-inf(d, 1); 0];
ub = inf(d+1, 1);

% 调用quadprog求解
opts = optimset('Display', 'off');
x = quadprog(H, f, A, b, [], [], lb, ub, [], opts);

W = x(1:d);
b = x(d+1);


end