clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/coil-20-proc/';%path：path for picture
Data_struct=dir([path,'*.png']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=71;%(cycle) picutre/group
ALL=5;%(get) ALL group
Tr=20;%Tr：trainData:(Tr)/group
Te=10;%Te：testData:(Te)/group
%Te+Tr<=cylce
k_remain=40;%in iter the dimension retained
iter=5;%iter time
k=5;%KNN in MFA,LPP
K=3;%KNN to calcuate the accuract rate
n=400;%to calculate the accuracy,from 1 to n the dimension you want to retain 
choose="PCA";%["PCA","LDA","LPP","MFA"]
picture_resize=[50,40];%resize the picture

%operation area
%pivture imread
[all_label,allData]=function_picture_import(cycle,ALL,Data_struct,path,picture_resize);

%classify the train and test

% [trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
[trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);

%algorithm area

if choose=="PCA"
    [~,U_PCA]=function_PCA(trainData);
    U_RIPCA=function_RIPCA(trainData,U_PCA,k_remain,iter);
elseif choose=="LDA"
    U_LDA=function_LDA(trainData,Tr,ALL);
    U_RILDA=function_RILDA(trainData,Tr,ALL,U_LDA,k_remain,iter);
elseif choose=="LPP"
    U_LPP=function_LPP(trainData,k);
    U_RILPP=function_RILPP(trainData,U_LPP,k,k_remain,iter);
elseif choose=="MFA"
    U_MFA=function_MFA(trainData,train_label,k);
    U_RIMFA=function_RIMFA(trainData,train_label,U_MFA,k,k_remain,iter);
else
    fprintf("error!\n");
    return;
end

%compare RI and normal algorithm
if choose=="PCA"
    U1=U_PCA;
    U2=U_RIPCA;
elseif choose=="LDA"
    U1=U_LDA;
    U2=U_RILDA;
elseif choose=="LPP"
    U1=U_LPP;
    U2=U_RILPP;
elseif choose=="MFA"
    U1=U_MFA;
    U2=U_RIMFA;
end

correct_1=zeros(1,n);
correct_2=zeros(1,n);

fprintf('accuracy calculating\n');

for i=1:n
    final_testData=U1(1:i,:)*testData;
    final_trainData=U1(1:i,:)*trainData;
    correct_1(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
    final_testData=U2(1:i,:)*testData;
    final_trainData=U2(1:i,:)*trainData;
    correct_2(i)=function_KNN(final_testData,final_trainData,test_label,train_label,K);
end

figure;
plot(1:n,correct_1,1:n,correct_2);
if choose=="PCA"
    legend({'PCA','RIPCA'},'Location','southwest')
elseif choose=="LDA"
    legend({'LDA','RILDA'},'Location','southwest')
elseif choose=="LPP"
    legend({'LPP','RILPP'},'Location','southwest')
elseif choose=="MFA"
    legend({'MFA','RIMFA'},'Location','southwest')
end

xlabel('dimension');
ylabel('accuracy');
title('accuracy in different dimension');
