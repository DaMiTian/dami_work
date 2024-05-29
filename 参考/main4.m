clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/coil-20-proc/';%path：path for picture
Data_struct=dir([path,'*.png']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=71;%(cycle) picutre/group
ALL=5;%(get) ALL group
Tr=60;%Tr：trainData:(Tr)/group
Te=10;%Te：testData:(Te)/group
%Te+Tr<=cylce
k_remain=40;%in iter the dimension retained
iter=5;%iter time
k=5;%KNN in MFA,LPP
K=3;%KNN to calcuate the accuract rate
n=100;%to calculate the accuracy,from 1 to n the dimension you want to retain 
picture_resize=[50,40];%resize the picture

%operation area
%pivture imread
[all_label,allData]=function_picture_import(cycle,ALL,Data_struct,path,picture_resize);

%classify the train and test

% [trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
[trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
%%
%NPE
clear;close all;clc;
%瑞士卷测试
N = 1000;
%Gaussian noise
noise = 0.001*randn(1,N);
%standard swiss roll data
tt = (3*pi/2)*(1+2*rand(1,N));   
height = 21*rand(1,N);
X = [(tt+ noise).*cos(tt); height; (tt+ noise).*sin(tt)];
%show the picture
point_size = 20;
figure;
scatter3(X(1,:),X(2,:),X(3,:), point_size,tt,'filled');
U_LPP=function_LPP(X,50);
X_LPP=U_LPP(1:2,:)*X;
U_NPE=function_NPE(X,50);
X_NPE=U_NPE(1:2,:)*X;
figure;
scatter(X_LPP(1,:),X_LPP(2,:), point_size,tt,'filled');
title('LPP');
figure;
scatter(X_NPE(1,:),X_NPE(2,:), point_size,tt,'filled');
title('NPE');
X_NPE=U_NPE(2:3,:)*X;
figure;
scatter(X_NPE(1,:),X_NPE(2,:), point_size,tt,'filled');
title('NPE_r');
%%
U_LPP=function_LPP(allData,10);
U_NPE=function_NPE(allData,	0);
k_keep=150;
index=1;
correct1=[];
correct_2=[];
correct_3=[];
for i=10:10:k_keep
    final_testData=U_NPE(1:i,:)*testData;
    final_trainData=U_NPE(1:i,:)*trainData;
    correct_1(index)=function_KNN(final_testData,final_trainData,test_label,train_label,3);
    final_testData=U_LPP(1:i,:)*testData;
    final_trainData=U_LPP(1:i,:)*trainData;
    correct_2(index)=function_KNN(final_testData,final_trainData,test_label,train_label,3);
    final_testData=testData;
    final_trainData=trainData;
    correct_3(index)=function_KNN(final_testData,final_trainData,test_label,train_label,3);
    index=index+1;
end

figure;
plot(10:10:k_keep,correct_1,10:10:k_keep,correct_2,10:10:k_keep,correct_3);
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{NPE}','U_{LPP}','KNN'},'Location','southwest')
%%
%二维hard SVM可视化
random=unifrnd(-1,1,2,50);
X1=ones(2,50)+random;
X2=3*ones(2,50)+random;
X=[X1,X2];
y=[ones(1,50),-1*ones(1,50)];
[w,b]=function_SVM_hard(X,y);
figure;
scatter(X1(1,:),X1(2,:),'r');
hold on;
scatter(X2(1,:),X2(2,:),'b');
hold on;
xlim([0 4])
ylim([-2 6])
k=-w(1)./w(2);%将直线改写成斜截式便于作图
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2);
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2);
plot(xx,yy,'--')
title('hard support vector machine');
xlabel('dimension1');
ylabel('dimension2');
zlabel('dimension3');
legend('group1','group2','separating hyperplane');
%%
%三维hard SVM可视化
random=unifrnd(-1.5,1.5,3,50);
X1=ones(3,50)+random;
X2=3*ones(3,50)+random;
X=[X1,X2];
y=[ones(1,50),-1*ones(1,50)];
[w,b]=function_SVM_hard(X,y);
figure;
scatter3(X1(1,:),X1(2,:),X1(3,:),'r');
hold on;
scatter3(X2(1,:),X2(2,:),X2(3,:),'b');
hold on;
x_plot=linspace(0,4,20);
y_plot=linspace(0,4,20);
[X_plot,Y_plot]=meshgrid(x_plot,y_plot); 
Z_plot=-(w(1)*X_plot+w(2)*Y_plot+b)/w(3);
h=surf(X_plot,Y_plot,Z_plot);
set(h,'FaceAlpha',0.5);
title('hard support vector machine');
xlabel('dimension1');
ylabel('dimension2');
zlabel('dimension3');
% legend('group1','group2','separating hyperplane');
%%
%二维soft SVM可视化
random=unifrnd(-1.4,1.4,2,50);
X1=ones(2,50)+random;
X2=3*ones(2,50)+random;
X=[X1,X2];
y=[ones(1,50),-1*ones(1,50)];
[w,b]=function_SVM_soft(X,y,10);
figure;
scatter(X1(1,:),X1(2,:),'r');
hold on;
scatter(X2(1,:),X2(2,:),'b');
hold on;
k=-w(1)./w(2);%将直线改写成斜截式便于作图
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2);
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2);
plot(xx,yy,'--')
title('support vector machine');
xlabel('dimension1');
ylabel('dimension2');
legend('group1','group2','soft separating hyperplane');
%%
%三维soft SVM可视化
random=unifrnd(-1.5,1.5,3,50);
X1=ones(3,50)+random;
X2=3*ones(3,50)+random;
X=[X1,X2];
y=[ones(1,50),-1*ones(1,50)];
[w,b]=function_SVM_soft(X,y,10);
figure;
scatter3(X1(1,:),X1(2,:),X1(3,:),'r');
hold on;
scatter3(X2(1,:),X2(2,:),X2(3,:),'b');
hold on;
x_plot=linspace(0,4,20);
y_plot=linspace(0,4,20);
[X_plot,Y_plot] = meshgrid(x_plot,y_plot);
Z_plot=-(w(1)*X_plot+w(2)*Y_plot+b)/w(3);
h=surf(X_plot,Y_plot,Z_plot);
set(h,'FaceAlpha',0.5);
title('soft support vector machine');
xlabel('dimension1');
ylabel('dimension2');
zlabel('dimension3');
% legend('group1','group2','separating hyperplane');
%%
%人脸数据识别度分析
index_acc=1;
acc=[];err=[];
acc_knn=[];
% [U_1]=function_PCA(trainData);
% [U_1,~,~,~]=function_LDA(trainData,Tr,ALL);
% [U_1]=function_LPP(allData,50);
% [U_1]=function_NPE(trainData,40);
% [U_1]=function_MFA(trainData,train_label,50);
for k=35:5:160
    W=[];B=[];
    U=U_1(1:k,:);
    %计算对应的W和b
    Tr_data=U*trainData;
    Te_data=U*testData;
    if any(imag(U(:)))
        Tr_data=[real(Tr_data);imag(Tr_data)];
        Te_data=[real(Te_data);imag(Te_data)];
        fprintf('虚数:%d\n',k);
    end
    for i=1:ALL
        for j=i+1:ALL
            error=0;
            X=[function_group(Tr_data,Tr,i),function_group(Tr_data,Tr,j)];y=[ones(1,Tr),-1*ones(1,Tr)];
            [W_ij,b_ij]=function_SVM_hard(X,y);
            W=[W,W_ij];
            B=[B,b_ij];
            for p=1:size(X,2)
                if y(p)*(W_ij'*X(:,p)+b_ij)<0
                    error=error+1;
                end
            end
            if error
                fprintf('!');
            end
            err=[err,error];
        end
    end
    index=1;
    label_weight=zeros(length(test_label),ALL);
    % close;figure;
    for i=1:ALL
        for j=i+1:ALL
            W_ij=W(:,index);
            b_ij=B(:,index);
            index=index+1;
            %可视化
            %二维可视化（当SVM计算不止二维时，二位可视化效果差）
%             clf;
%             function_SVM_2d(function_group(Tr_data,Tr,i),function_group(Tr_data,Tr,j),W_ij,b_ij);
%             pause(3);
         
            for p=1:length(test_label)
                if W_ij'*(Te_data(:,p))+b_ij>0
                    label_weight(p,i)=label_weight(p,i)+1;
                else
                    label_weight(p,j)=label_weight(p,j)+1;
                end
            end
        end
    end
    
    error=0;
    test_label_pre=zeros(1,length(test_label));
    for p=1:length(test_label)
        [~,test_label_pre(p)]=max(label_weight(p,:));
        if test_label_pre(p)~=test_label(p)
            error=error+1;
        end
    end
    acc(index_acc)=(length(test_label)-error)/length(test_label);
    
    final_testData=U*testData;
    final_trainData=U*trainData;
    acc_knn(index_acc)=function_KNN(final_testData,final_trainData,test_label,train_label,3)/100;
    
    index_acc=index_acc+1;
end
figure;
plot(35:5:160,acc,'-*',35:5:160,acc_knn,'--');
xlabel('dimention');
ylabel('acc');
title('不同维度下LPP+SVM的识别正确率');
legend({'LPP+SVM','LPP+KNN(3)'},'Location','southwest')

%%
%kernel SVM人脸数据识别度分析
% [U_1,~,~,~]=function_LDA(trainData,Tr,ALL);
[U_1]=function_NPE(allData,5);
A=[];B=[];
U=U_1(1:10,:);
global sigma;
dist=0;
for p=1:ALL
    for i=1:Tr
        for j=1:Tr
            dist=dist+norm(U*trainData(:,(p-1)*Tr+i)-U*trainData(:,(p-1)*Tr+j))^2;
        end
    end
    dist=dist/(Tr^2);
end
dist=dist/ALL;
sigma=sqrt(dist);
%计算对应的W和b
for i=1:ALL
    for j=i+1:ALL
        X=[U*function_group(trainData,Tr,i),U*function_group(trainData,Tr,j)];y=[ones(1,Tr),-1*ones(1,Tr)];
        [a,b_ij]=function_SVM_kernel(X,y);
        A=[A,a];
        B=[B,b_ij];
    end
end
index=1;
label_weight=zeros(length(test_label),ALL);
for i=1:ALL
    for j=i+1:ALL
        alpha_ij=A(:,index);
        b_ij=B(:,index);
        index=index+1;
        for p=1:length(test_label)
            yy=0;
            for t=1:size(X,2)
                yy=yy+alpha_ij*kernel(U*testData(:,p),X(:,t))+b_ij;
            end
%             fprintf("%d\n",yy);
            if yy>0
                label_weight(p,i)=label_weight(p,i)+1;
            else
                label_weight(p,j)=label_weight(p,j)+1;
            end
        end
    end
end

