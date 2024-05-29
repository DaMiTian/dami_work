function [center,label]=function_k_means(data,k)
[m,n]=size(data);
center=zeros(m,k);
%初始化中心点
for i=1:k
    center(:,i)=data(:,ceil(rand()*n));
%     center(:,i)=data(:,i*5-1);
end
%更新距离
delta_center=inf;
new_center=zeros(m,k);
while(delta_center>0.01)
    Dist=dist_k();
    [~,label]=min(Dist);
    for belong=1:k
        find_label=find(label==belong);
        num=length(find_label);
        new_center(:,belong)=sum(data(:,find_label),2)/num;
    end
    delta_center=norm(new_center-center,2);
    center=new_center;
end

    function Dist=dist_k()
        Dist=zeros(k,n);
        for ii=1:k
            for jj=1:n
                Dist(ii,jj)=dist(center(:,ii),data(:,jj));
            end
        end
    end
    function dis=dist(d1,d2)
        dis=sqrt(norm(d1-d2,2));
    end
end