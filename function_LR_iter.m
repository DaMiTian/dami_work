function [W,i]=function_LR_iter(Y,X,iter,W)
% fprintf("delta_Y:%f,iter:%d\n",norm(Y-W*X,2),0);
for i=1:iter
%     fprintf('iter:%d\n',i);
    d_W=(W*X-Y)*X';
    lr=norm(d_W,2)^2/norm(d_W*X)^2;
    W_tmp=W-lr.*d_W;
    delta_W(i)=norm(W-W_tmp,2)/norm(W_tmp,2);
    delta_Y(i)=norm(Y-W_tmp*X,2);
    %fprintf("delta_Y:%f\n",delta_Y(i));
    %fprintf("delta_W:%f\n",delta_W(i));
    W=W_tmp;
    if i>1
        if delta_W(i)<0.01 && delta_Y(i)<delta_Y(i-1)
%             fprintf('itertime:%d,end!\n',i);
            break;
        end
    end
    if i>=iter
%         fprintf('iter end!\n');
        break
    end
end
end