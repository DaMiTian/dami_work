%计算样本矩阵Y
function Y=label_cal(trainData,U,Tr,ALL,k_keep)
Y=zeros(k_keep,size(trainData,2));
for i=1:ALL
    meanData=zeros(size(trainData,1),1);
    for j=(i-1)*Tr+1:i*Tr
        meanData=meanData+trainData(:,j);
    end
    for j=(i-1)*Tr+1:i*Tr
        Y(:,j)=U(1:k_keep,:)*meanData;
    end
end
end