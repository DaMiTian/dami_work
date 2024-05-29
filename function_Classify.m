%classify

function [trainData,train_label,testData,test_label]=function_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path)
A=imread([path,Data_struct(1).name]);%imread a picture
A=double(A);
A=A(:);
[m,~]=size(A);%get the size of picture
trainData=zeros(m,0);
train_label=zeros(1,0);

for i=1:ALL
    for j=(i-1)*cycle+1:(i-1)*cycle+Tr
        trainData=[trainData,allData(:,j)];
        train_label=[train_label,all_label(1,j)];
    end
end

testData=zeros(m,0);
test_label=zeros(1,ALL*Te);
x=1;
for i=1:ALL
    for j=(i-1)*cycle+Tr+1:(i-1)*cycle+Tr+Te
        testData=[testData,allData(:,j)];
        test_label(1,x)=all_label(1,j);
        x=x+1;
    end
end
end