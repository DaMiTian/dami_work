clc;clear;

%Data initialization area
path='C:/Users/12505/Desktop/machine_learning/dataset/Yale_face10080/';%path：path for picture
Data_struct=dir([path,'*.bmp']);%['*,tif','*.bmp']
%Data_struct：store ipcture information
cycle=11;%(cycle) picutre/group
ALL=5;%(get) ALL group
Tr=6;%Tr：trainData:(Tr)/group
Te=5;%Te：testData:(Te)/group
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

[trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
% [trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path);
%%
%二维k_means验证
clc;
random=unifrnd(-1,1,2,50);
X1=ones(2,50)+random;
X2=3*ones(2,50)+random;
X3=[2*ones(1,50);-1*ones(1,50)]+random;
X=[X1,X2,X3];
k=3;
[center,label]=function_k_means(X,3);
figure;
hold on;
for i=1:k
    find_label=find(label==i);
    scatter(X(1,find_label),X(2,find_label));
end
scatter(center(1,:),center(2,:),'*');
title('二维k means验证');
%%
%三维k_means验证
random=unifrnd(-1,1,3,50);
X1=ones(3,50)+random;
X2=3*ones(3,50)+random;
X3=[2*ones(1,50);-1*ones(1,50);-1*ones(1,50)]+random;
X=[X1,X2,X3];
k=3;
[center,label]=function_k_means(X,3);
figure;
for i=1:k
    find_label=find(label==i);
    scatter3(X(1,find_label),X(2,find_label),X(3,find_label));
    hold on;
end
scatter3(center(1,:),center(2,:),center(3,:),'*');
title('三维k means验证');
%%
clc;
error=0;
for i=1:k
    l=find(label==i);
    aver_i=sum(l)/length(l);
    if aver_i<=50
        error=error+length(find(l>50));
    elseif aver_i<=100
        error=error+length(find(l<50))+length(find(l>100));   
    else
        error=error+length(find(l<=100));       
    end
end
error=1-error/150;
fprintf('正确率：%f\n',error);
%%
%PCA降维后聚类
clc;
[U_PCA]=function_PCA(trainData);
U_PCA=U_PCA(1:3,:);
Data=U_PCA*testData;
[center,label]=function_k_means(Data,ALL);
acc=acc_cal(label,test_label,k);
%%
%LDA降维后聚类
clc;
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);
U_LDA=U_LDA(1:3,:);
Data=U_LDA*testData;
[center,label]=function_k_means(Data,ALL);
acc=acc_cal(label,test_label,k);
%%
%三维
img = cell(100,1); % 预分配一个100×1的cell数组
img_x=250;img_y=200;
for i=1:size(Data,2)
    img_i=reshape(trainData(:,i),[50,40]);
    img{i}=imresize(img_i,[img_x,img_y]);
end

figure;
for i=1:k
    find_label=find(label==i);
    scatter3(Data(1,find_label),Data(2,find_label),Data(3,find_label));
    hold on;
end

scatter3(center(1,:),center(2,:),center(3,:),'*');
title('LDA+K means分类结果');
%%
%二维图点对应
clc;
img = cell(100,1); % 预分配一个100×1的cell数组
img_x=250;img_y=200;
for i=1:size(Data,2)
    img_i=reshape(testData(:,i),[50,40]);
    img{i}=imresize(img_i,[img_x,img_y]);
end
figure;
hold on;
hFig = gcf;
for i = 1:size(Data,2)
    x = Data(1, i);
    y = Data(2, i);
    imge = flipud(img{i});
    image([x x+img_x], [y y+img_y], imge);
    colormap(gray);
end
for i=1:k
    find_label=find(label==i);
    scatter(Data(1,find_label),Data(2,find_label));
end

title('PCA k=5');
%%
[U_LDA,~,~,~]=function_LDA(trainData,Tr,ALL);
[U_PCA]=function_PCA(trainData);
acc_PCA=zeros(5,15);acc_LDA=zeros(5,15);
for k=3:7
    for dimention=10:10:150
        for t=1:10
            t_data=U_LDA(1:dimention,:)*testData;
            [~,label]=function_k_means(t_data,k);
            acc_LDA(k-2,dimention/10)=acc_LDA(k-2,dimention/10)+acc_cal(label,test_label,k);

            t_data=U_PCA(1:dimention,:)*testData;
            [~,label]=function_k_means(t_data,k);
            acc_PCA(k-2,dimention/10)=acc_PCA(k-2,dimention/10)+acc_cal(label,test_label,k);
        end
    end
end
acc_PCA=acc_PCA./10;acc_LDA=acc_LDA./10;

figure;
x = 10:10:150;
y = 3:7;
[X,Y] = meshgrid(x,y);
surf(X, Y, acc_LDA)
title('LDA算法下k means');
xlabel('保留维度数')
ylabel('k值')
zlabel('正确率')

figure;
x = 10:10:150;
y = 3:7;
[X,Y] = meshgrid(x,y);
surf(X, Y, acc_PCA)
title('PCA算法下k means');
xlabel('保留维度数')
ylabel('k值')
zlabel('正确率')

%%
function acc=acc_cal(label,test_label,k)
error=0;
for i=1:k
    l=find(label==i);
    most_i=mode(test_label(l));%找到第k类中真实类别最多的
    error=error+length(find(test_label(l)~=most_i));
end
acc=(length(label)-error)/length(label);
end





