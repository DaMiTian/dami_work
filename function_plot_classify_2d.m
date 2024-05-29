function function_plot_classify_2d(Y,i,j,color)
x=Y(1,:);
y=Y(2,:);

x1=x(:,i:j);
y1=y(:,i:j);

scatter(x1,y1,color);
end