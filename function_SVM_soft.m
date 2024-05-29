function [w,b]=function_SVM_soft(X,y,C)
[~,n]=size(X);
h=zeros(n,n);
for i=1:n
    for j=1:n
        h(i,j)=X(:,i)'*X(:,j)*y(i)*y(j);
    end
end
f=-1*ones(n,1);%目标函数的f

%等式约束 aeq*x=beq
aeq=y;
beq=zeros(1,1);
%自变量约束lb(i)<=x(i)<=ub
lb=zeros(n,1);
ub=C*ones(n,1);
[a,~]=quadprog(h,f,[],[],aeq,beq,lb,ub);%二次规划问题

for i=1:length(a)
    if a(i)<1e-5
        a(i)=0;
    end
    if abs(a(i)-C)<1e-5
        a(i)=C;
    end
end

w=0;%系数矩阵
u=0;
ff=find(a~=0);
ff_2=find(a==C);
ptr=ff(1);
for i=1:size(ff,1)
    flag=1;
    for j=1:size(ff_2,1)
        if ff(i)==ff_2(j)
            flag=0;
            break;
        end
    end
    if flag==1
        ptr=ff(i);
        break;
    end
end
for i=1:n%关键系数数求解
    w=w+a(i)*y(i)*X(:,i);
    u=u+a(i)*y(i)*(X(:,i)'*X(:,ptr));
end
b=y(ptr)-u;
end