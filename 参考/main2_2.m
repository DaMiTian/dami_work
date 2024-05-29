clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/Yale_face10080/';%path：path for picture
Data_struct=dir([path,'*.bmp']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=11;%(cycle) picutre/group
ALL=15;%(get) ALL group
Tr=6;%Tr：trainData:(Tr)/group
Te=4;%Te：testData:(Te)/group
%Te+Tr<=cylce
iter=5;%iter time
picture_resize=[50,40];%resize the picture

%operation area
%pivture imread
[all_label,allData]=function_picture_import(cycle,ALL,Data_struct,path,picture_resize);

%classify the train and test

% [trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
[trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);

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
    Y=label_cal(trainData,U_PCA,Tr,ALL,k_keep);
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
