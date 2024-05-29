%LDA

function [V1,V2,V3,V4]=function_LDA(allData,cycle,ALL)
V1=zeros(1);
V2=zeros(1);
V3=zeros(1);
V4=zeros(1);
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

% [P,D]=eig(Sb);
% for i=1:size(Sb,1)
%     D(i,i)=sqrt(D(i,i));
% end
% Sb_=P*D*pinv(P);
[V1,D1]=eig(pinv(Sw)*Sb);
% [V2,D2]=eig(pinv(Sw+eye(size(Sw,1))*10^(-6))*Sb);
% [V3,D3]=eig(Sb*pinv(Sw));
% [V4,D4]=eig(Sb-Sw);
[~,order]=sort(diag(D1),'descend');
V1=V1(:,order)';
% [~,order]=sort(diag(D2),'descend');
% V2=V2(:,order)';
% [~,order]=sort(diag(D3),'descend');
% V3=V3(:,order)';
% [~,order]=sort(diag(D4),'descend');
% V4=V4(:,order)';
end