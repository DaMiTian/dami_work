function function_SVM_2d(X1,X2,w,b)
scatter(X1(1,:),X1(2,:),'r');
hold on;
scatter(X2(1,:),X2(2,:),'b');
hold on;
k=-w(1)./w(2);%将直线改写成斜截式便于作图
bb=-b./w(2);
x_left=min([X1(1,:),X2(1,:)]);
x_right=max([X1(1,:),X2(1,:)]);
line([x_left x_right],[k.*x_left+bb k.*x_right+bb],'linestyle','-','color','m');
hold on;
line([x_left x_right],[k.*x_left+bb+1./w(2) k.*x_right+bb+1./w(2)],'linestyle','--','color','c');
hold on;
line([x_left x_right],[k.*x_left+bb-1./w(2) k.*x_right+bb-1./w(2)],'linestyle','--','color','c');
title('support vector machine');
xlabel('dimension1');
ylabel('dimension2');
end