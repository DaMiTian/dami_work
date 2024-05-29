%classify

function [trainData,train_label,testData,test_label]=function_random_Classify(Tr,Te,ALL,cycle,allData,all_label,Data_struct,path)
A=imread([path,Data_struct(1).name]);
A=double(A);
A=A(:);
[m,~]=size(A);
trainData=zeros(m,0);
train_label=zeros(1,0);

testData=zeros(m,0);
test_label=zeros(1,ALL*Te);
x_tr=1;
x_te=1;
for i=1:ALL
    ii=(i-1)*cycle+1:i*cycle;
    ii=ii(randperm(length(ii)));
    for j=1:Tr
        trainData=[trainData,allData(:,ii(j))];
        train_label(1,x_tr)=all_label(1,ii(j));
        x_tr=x_tr+1;
    end
    for j=Tr+1:Tr+Te
        testData=[testData,allData(:,ii(j))];
        test_label(1,x_te)=all_label(1,ii(j));
        x_te=x_te+1;
    end
end
end