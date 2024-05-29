%算法测试验证
%初始化

%%
%半嵌入模型
clc;clear;close;

n1=60;
n2=80;
X1=20+20*rand(1,n1);
X1=[X1;30+30*rand(1,n1)];
scatter(X1(1,:),X1(2,:),'r');


X2=10+14*rand(1,n2/2);
X2=[X2;5+40*rand(1,n2/2)];
X2=[X2,[20+20*rand(1,n2/2);2+25*rand(1,n2/2)]];
hold on;
scatter(X2(1,:),X2(2,:),'b');
%%
%全嵌入模型
clc;clear;close;

n1=60;
n2=100;
X1=[];X2=[];
r=20*rand(1,n1);
theta=2*pi*rand(1,n1);
X1=[r.*sin(theta);r.*cos(theta)];
scatter(X1(1,:),X1(2,:),'r');

r=18+10*rand(1,n2);
theta=2*pi*rand(1,n2);
X2=[r.*sin(theta);r.*cos(theta)];
hold on;
scatter(X2(1,:),X2(2,:),'b');
%%
k=3;
K_all=[];
x1_in=zeros(1,n1);
x2_in=zeros(1,n2);
for i=1:n1
    [kp,K]=function_manifold_KNN_point(X2,X1(:,i),k);
%     hold on;
%     scatter(kp(1,:),kp(2,:),'p');
%     xlim([0 60]);
%     ylim([0 60]);
    for j=1:k
        x2_in(K(j))=1;
    end
end
for i=1:n2
    [~,K]=function_manifold_KNN_point(X1,X2(:,i),k);
    for j=1:k
        x1_in(K(j))=1;
    end
end
%计算在边缘的点的坐标
X1_in=zeros(2,0);
for i=1:n1
    if x1_in(i)
        X1_in=[X1_in,X1(:,i)];
    end
end

X2_in=zeros(2,0);
for i=1:n2
    if x2_in(i)
        X2_in=[X2_in,X2(:,i)];
    end
end

X_all=[X1_in,X2_in];
%边缘点对应kk近邻
X=zeros(2,0);
kk=1;
for i=1:length(X1_in)
    [~,K2]=function_manifold_KNN_point(X2_in,X1_in(:,i),kk);
    K_all=[K_all,K2'+length(X1_in)];
end
for i=1:length(X2_in)
    [~,K2]=function_manifold_KNN_point(X1_in,X2_in(:,i),kk);
    K_all=[K_all,K2'];
end

%分成簇
%初始化，所有点都是单独成簇
X_all=[X1_in,X2_in];
len=length(X_all);
flag=1;
select_X=zeros(1,len);
for i=1:len
    select_X(i)=i;
end
while flag
    flag=0;
    for i=1:len
        for j=1:kk
            if select_X(K_all(j,i))~=select_X(i)
                flag=1;
                if select_X(K_all(j,i))<select_X(i)
                    select_X(i)=select_X(K_all(j,i));
                else
                    select_X(K_all(j,i))=select_X(i);
                end
            end
        end
    end
end

num=1;
for i=1:len
    if select_X(i)
        f=find(select_X==select_X(i));
        X_SVM=[];
        Y_SVM=[];
        if length(f)<6
            continue;
        end
        for j=1:length(f)
            X_SVM=[X_SVM,X_all(:,f(j))];
            if f(j)>length(X1_in)
                Y_SVM=[Y_SVM,-1];
            else
                Y_SVM=[Y_SVM,1];
            end
        end
        [w,b]=function_SVM_soft(X_SVM,Y_SVM,10);
        k=-w(1)./w(2);%将直线改写成斜截式便于作图
        x_left=min(X_SVM(1,:))-10;
        x_right=max(X_SVM(1,:))+10;
        bb=-b./w(2);
        hold on;
        xx=x_left:0.1:x_right;
        yy=k.*xx+bb;
        plot(xx,yy,'-');
        num=num+1;
        for j=1:length(f)
            select_X(f(j))=0;
        end
    end
    xlim([0 60]);
    ylim([0 60]);
end











% hold on;
% scatter(X(1,:),X(2,:),'p');
% 
% xlim([0 60]);
% ylim([0 60]);