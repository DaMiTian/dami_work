%导入图片与标签

function [all_label,allData]=function_picture_import(cycle,ALL,Data_struct,path,resize)
A=imread([path,Data_struct(1).name]);
A=double(A);
A=imresize(A,resize);
%A=rgb2gray(A);
A=A(:);%转换成列矩阵
[m,n]=size(A);%确定图片格式
all_label=zeros(1,ALL*cycle);%初始化
allData=zeros(m*n,0);%初始化
x=1;%中间变量
for j=1:ALL
    for i=(j-1)*cycle+1:(j-1)*cycle+cycle%读入第一张图片
        A=imread([path,Data_struct(i).name]);
        A=double(A);%转换成矩阵形式
        A=imresize(A,resize);
        A=A(:);%转换成列矩阵
        allData=[allData,A];
        all_label(1,x)=j;
        x=x+1;
    end
end
end