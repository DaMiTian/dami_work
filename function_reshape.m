function reshape_data=function_reshape(U,data,k,picture_resize)
    reshape_data=U(1:k,:)'*U(1:k,:)*data;
    reshape_data=reshape(reshape_data,picture_resize);
end

