%LDA

function [V1]=function_LDA_improve(allData,cycle,ALL,iter)
V1=zeros(1);

Sw=zeros(size(allData,1),size(allData,1));
for i=1:ALL
    Sw_tmp=zeros(size(allData,1),size(allData,1));
    mean_i=mean(allData(:,(i-1)*cycle+1:i*cycle),2);
    for j=(i-1)*cycle+1:i*cycle
        Sw_tmp=Sw_tmp+allData(:,j)*allData(:,j)';
    end
    Sw=Sw+Sw_tmp/cycle-(mean_i*mean_i');
end

Sb=zeros(size(allData,1),size(allData,1));
mean_all=zeros(size(allData,1),0);
for i=1:ALL
    mean_all=[mean_all,mean(allData(:,(i-1)*cycle+1:i*cycle),2)];
end
for i=1:ALL
    for j=1:ALL
        Sb=Sb+(mean_all(:,i)-mean_all(:,j))*(mean_all(:,i)-mean_all(:,j))';
    end
end
[V1,D1]=eig(pinv(Sw)*Sb);
[~,order]=sort(diag(D1),'descend');
V1=V1(:,order)';
fprintf('初始化完成，进入迭代');

for i=1:iter
    D=zeros(size(allData,1),size(allData,1));
    for j=1:size(allData,1)
        D(j,j)=norm(V1(j,:),2);
    end
    [new_V1,D1]=eig(pinv(Sw)*(Sb+D));
    [~,order]=sort(diag(D1),'descend');
    new_V1=new_V1(:,order)';
    Delta_U=norm(new_V1-V1,2);
    fprintf('iter:%d Delta_U:%f',i,Delta_U);
    if Delta_U<1
        break;
    end
    V1=new_V1;
end

end