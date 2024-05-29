function function_plot_3d(Y,i,j,color)
x=Y(1,:);
y=Y(2,:);
z=Y(3,:);

x1=x(:,i:j);
y1=y(:,i:j);
z1=z(:,i:j);
scatter3(x1,y1,z1,color);
end