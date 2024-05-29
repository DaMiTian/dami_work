clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/AR_Gray_50by40/';%path：path for picture
Data_struct=dir([path,'*.tif']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=26;%(cycle) picutre/group
ALL=20;%(get) ALL group
Tr=16;%Tr：trainData:(Tr)/group
Te=10;%Te：testData:(Te)/group
%Te+Tr<=cylce
k_remain=40;%in iter the dimension retained
iter=5;%iter time
k=5;%KNN in MFA,LPP
K=1;%KNN to calcuate the accuract rate
n=100;%to calculate the accuracy,from 1 to n the dimension you want to retain 
picture_resize=[50,40];%resize the picture

%operation area
%pivture imread
[all_label,allData]=function_picture_import(cycle,ALL,Data_struct,path,picture_resize);

%classify the train and test

% [trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
[trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
%%
index=1;
[U]=function_PCA(trainData);
correct_1=zeros(1,80);
correct_2=zeros(1,80);
time_1=zeros(1,80);
time_2=zeros(1,80);
for k_keep=20:200
    Y=label_cal(trainData,U,Tr,ALL,k_keep);
    tic;
    W_LR=function_LR(Y,trainData);
    %fprintf('delta_Y=%f\n',norm(Y-W*trainData,2));
    time_1(index)=toc;
    tic;
    W_iter=function_LR_iter(Y,trainData,100,U(1:k_keep,:));
    time_2(index)=toc;
    final_testData=W_LR*testData;
    final_trainData=W_LR*trainData;
    correct_1(index)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=W_iter*testData;
    final_trainData=W_iter*trainData;
    correct_2(index)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    index=index+1;
end

figure;
plot(20:k_keep,correct_1,20:k_keep,correct_2);
xlabel('dimension');
ylabel('accuracy');
title('不同维度准确度');
legend({'最小二乘法','梯度下降法'},'Location','southwest')

figure;
plot(20:k_keep,time_1,20:k_keep,time_2);
xlabel('dimension');
ylabel('accuracy');
title('不同方法运行时间');
legend({'最小二乘法时间','梯度下降法时间'},'Location','southwest')
%%
%RI_regression
k_keep=40;
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);
Y=label_cal(trainData,U_LDA,Tr,ALL,k_keep);

% U_RI=RI(trainData,U,Y,k_keep,iter);
[U_RI,~]=function_RI_regression(trainData,U_LDA,Y,k_keep,1);
[U_PCA]=function_PCA(trainData);
U1=U_RI';
U1=U1(:,1:size(U_LDA,2));
K=5;

for i=1:k_keep
    Y=label_cal(trainData,U_PCA,Tr,ALL,i);
    W_LR=function_LR(Y,trainData);
    W_iter=function_LR_iter(Y,trainData,100,U_PCA(1:k_keep,:));
    final_testData=U1(1:i,:)*testData;
    final_trainData=U1(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U2=W_LR;
    final_testData=U2*testData;
    final_trainData=U2*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U3=W_iter;
    final_testData=U3*testData;
    final_trainData=U3*trainData;
    correct_3(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:k_keep,correct_1,1:k_keep,correct_2,1:k_keep,correct_3);
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{RFS}','U_{LR}','{U_LR_iter}'},'Location','southwest')

%%
%UDFS
[U_Unsupervised_Learning_LR]=function_Unsupervised_Learning_LR(trainData,5,10,3);
[U_PCA]=function_PCA(trainData);

index=1;
for k_keep=100:5:400
    Y=label_cal(trainData,U_PCA,Tr,ALL,k_keep);
    W_LR=function_LR(Y,trainData);
    W_iter=function_LR_iter(Y,trainData,100,U_PCA(1:k_keep,:));
    U1=U_Unsupervised_Learning_LR(1:k_keep,:);
    final_testData=U1*testData;
    final_trainData=U1*trainData;
    correct_1(index)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U2=W_LR;
    final_testData=U2*testData;
    final_trainData=U2*trainData;
    correct_2(index)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U3=W_iter;
    final_testData=U3*testData;
    final_trainData=U3*trainData;
    correct_3(index)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    index=index+1;
end

figure;
plot(100:5:400,correct_1,100:5:400,correct_2,100:5:400,correct_3);
xlabel('dimension');
ylabel('accuracy');
title('Unsupervised Learning LR');
legend({'UDFS','线性回归求导','线性回归梯度下降'},'Location','southwest')

%%
%SRLDA
k_keep=40;
U_SRLDA=function_SRLDA(trainData,k_keep,Tr,ALL);
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);

U1=U_SRLDA;

for i=1:k_keep
    Y=label_cal(trainData,U_LDA,Tr,ALL,i);
    W_LR=function_LR(Y,trainData);
    W_iter=function_LR_iter(Y,trainData,100,U_LDA(1:k_keep,:));
    final_testData=U1(1:i,:)*testData;
    final_trainData=U1(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U2=W_LR;
    final_testData=U2*testData;
    final_trainData=U2*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U3=W_iter;
    final_testData=U3*testData;
    final_trainData=U3*trainData;
    correct_3(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:k_keep,correct_1,1:k_keep,correct_2,1:k_keep,correct_3);
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{SRLDA}','U_{LR}','{U_LR_iter}'},'Location','southwest')

%%
%二维可视化
figure;
finalData=W_iter(1:2,:)*allData;
function_plot_classify_2d(finalData,1,10,'r');
hold on;
function_plot_classify_2d(finalData,11,20,'b');
hold on;
function_plot_classify_2d(finalData,21,30,'k');
hold on;
function_plot_classify_2d(finalData,31,40,'g');
title('LR iter 2d');

%%
%人脸重构
figure;
subplot(2,4,1);
for i=10:10:80
    picture_Data=function_reshape(W_LR,allData(:,i),80,picture_resize);
    subplot(2,4,i/10);
    imshow(picture_Data,[]);
end
%%
%LR_improve

k_keep=20;
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);

for i=1:k_keep
    Y=label_cal(trainData,U_LDA,Tr,ALL,i);
    W_LR=function_LR(Y,trainData);
    W_iter=function_LR_iter(Y,trainData,100,U_LDA(1:i,:));
    U_LR_improve=function_LR_improve(Y,trainData);
    U1=U_LR_improve;
    final_testData=U1*testData;
    final_trainData=U1*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U2=W_LR;
    final_testData=U2*testData;
    final_trainData=U2*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    U3=W_iter;
    final_testData=U3*testData;
    final_trainData=U3*trainData;
    correct_3(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:k_keep,correct_1,'*',1:k_keep,correct_2,'-',1:k_keep,correct_3);
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{LR improve}','U_{LR}','{U LR iter}'},'Location','southwest')
