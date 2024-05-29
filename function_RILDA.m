%RILDA

function U=function_RILDA(allData,cycle,ALL,U_init,k_remain,time)
U=U_init;
mean_all=zeros(size(allData,1),ALL);
mean_aver=mean(allData,2);
for i=1:ALL
    mean_all(:,i)=mean(allData(:,(i-1)*cycle+1:i*cycle),2);
end
Xb=zeros(size(allData,1),ALL);
for i=1:ALL
    Xb(:,i)=mean_all(:,i)-mean_aver;
end
Xw=zeros(size(allData,1),ALL*cycle);
for i=1:ALL
    for j=(i-1)*cycle+1:i*cycle
        Xw(:,j)=allData(:,j)-mean_all(:,i);
    end
end

for iter=1:time
    fprintf('iter:%d\n',iter);
    Dw=zeros(ALL*cycle,ALL*cycle);
    Db=zeros(ALL,ALL);
    for j=1:ALL
        Db(j,j)=1/sqrt((U(1:k_remain,:)*Xb(:,j))'*(U(1:k_remain,:)*Xb(:,j)));
    end
    for j=1:ALL*cycle
        Dw(j,j)=1/sqrt((U(1:k_remain,:)*Xw(:,j))'*(U(1:k_remain,:)*Xw(:,j)));
    end
    [V,D]=eig(pinv(Xb*Db*Xb')*Xw*Dw*Xw');
    d=diag(D);
    [~,order]=sort(d,'descend');
    V=V(:,order);
    V=V';
    delta_U=norm(V-U,2);
     fprintf("iter:%d,delta_U=%d\n",iter,delta_U);
    if iter>5
        if delta_U<1
            fprintf('break,%d\n',iter);
            break;
        end
    end
    U=V;
end
end