clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/AR_Gray_50by40/';%path：path for picture
Data_struct=dir([path,'*.tif']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=26;%(cycle) picutre/group
ALL=100;%(get) ALL group
Tr=10;%Tr：trainData:(Tr)/group
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
%LDA
[V1,V2,V3,V4]=function_LDA(trainData,Tr,ALL);
%%
%PCA
[U_PCA]=function_PCA(trainData);
%%
%不同LDA比较
for i=1:n
    final_testData=V1(1:i,:)*testData;
    final_trainData=V1(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=V2(1:i,:)*testData;
    final_trainData=V2(1:i,:)*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=V3(1:i,:)*testData;
    final_trainData=V3(1:i,:)*trainData;
    correct_3(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=V4(1:i,:)*testData;
    final_trainData=V4(1:i,:)*trainData;
    correct_4(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:n,correct_1(1:n),1:n,correct_2(1:n),1:n,correct_3(1:n),1:n,correct_4(1:n));
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'S_{w}^{-1}*S_b','(S_{W}+I*10^{-6})^{-1}*S_b','S_b*S_{w}^-1','S_b - S_w'},'Location','southwest')


%%
%PCA与LDA比较
U1=U_PCA;
U2=V1;

for i=1:n
    final_testData=U1(1:i,:)*testData;
    final_trainData=U1(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=U2(1:i,:)*testData;
    final_trainData=U2(1:i,:)*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:n,correct_1(1:n),1:n,correct_2(1:n));
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{PCA}','U_{LDA}'},'Location','southwest')
%%
% 还原数据
figure;
subplot(2,4,1);
imshow(reshape(allData(:,40),picture_resize),[]);
for i=2:1:8
    picture_Data=function_reshape(U_PCA,allData(:,40),i,picture_resize);
    subplot(2,4,i);
    imshow(picture_Data,[]);
end
%%
figure;
finalData=U_PCA(1:3,:)*allData;
function_plot_3d(finalData,1,7,'r');
hold on;
function_plot_3d(finalData,8,14,'b');
hold on;
function_plot_3d(finalData,15,21,'k');
hold on;
function_plot_3d(finalData,22,28,'g');
title('PCA 3d');

figure;
finalData=U_LDA(1:3,:)*allData;
function_plot_3d(finalData,1,7,'r');
hold on;
function_plot_3d(finalData,8,14,'b');
hold on;
function_plot_3d(finalData,15,21,'k');
hold on;
function_plot_3d(finalData,22,28,'g');
title('LDA 3d');

figure;
finalData=U_PCA(1:2,:)*allData;
function_plot_classify_2d(finalData,1,7,'r');
hold on;
function_plot_classify_2d(finalData,8,14,'b');
hold on;
function_plot_classify_2d(finalData,15,21,'k');
hold on;
function_plot_classify_2d(finalData,22,28,'g');
title('PCA 2d');

figure;
finalData=U_LDA(1:2,:)*allData;
function_plot_classify_2d(finalData,1,7,'r');
hold on;
function_plot_classify_2d(finalData,8,14,'b');
hold on;
function_plot_classify_2d(finalData,15,21,'k');
hold on;
function_plot_classify_2d(finalData,22,28,'g');
title('LDA 2d');
%%
%改进LDA
[U_LDA_improve]=function_LDA_improve(allData,cycle,ALL,5);
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);

for i=1:n
    final_testData=U_LDA(1:i,:)*testData;
    final_trainData=U_LDA(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=U_LDA_improve(1:i,:)*testData;
    final_trainData=U_LDA_improve(1:i,:)*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:n,correct_1(1:n),1:n,correct_2(1:n));
xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
legend({'U_{LDA}','U_{LDA_improve}'},'Location','southwest')