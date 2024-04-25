function [X,Y,error_list] = splitadmm20240119(A,G,eta,beta,iter,X1,Y1)

[n,k] = size(G);
epsi = 1e-5;
en = ones(n,1);
ek = ones(k,1);
Lambda = zeros(n,k);
maxiter = iter;
iter = 0;
X = X1;
Y = Y1;
%fprintf('%i\t%f\t%f\n',iter,norm(X-Y,'fro'),obj(X,Y))
error_list = [];
saved_obj = [];
fprintf('%f,%f\n',beta,eta)
for iter = 1:maxiter
tic;
beta_scale = beta * norm(X-Y,"fro")^2;
eta_scale = eta * sum(Y,'all');
obj_value = sum(Y.*(A*X),'all')/2+sum(G.*X,'all')+sum(Lambda.*(X-Y),'all');
fprintf('%i,%e,%e,%e\n',iter,beta_scale,eta_scale,obj_value)
%update X
X = Y-(Lambda+A*Y/2+G)/beta;
B = X;
%X = X-1/n*en*en'*X+1/k*(en-X*ek+1/n*en*en'*X*ek)*ek'; %计算时间久
% 优化版
eX = sum(X,1);
eeX = repmat(eX,n,1);
Xe = sum(X,2);
X_p2 = 1/k*(en-Xe +1/n* sum(eeX,2)  );
X = X-1/n  *  eeX  +  repmat(X_p2,1,k);
fprintf('%i\t%f\t%f\n',iter,norm(X-Y,'fro'),obj(X,Y))

temp_Y = Y;

%update Y
Y_ = X+(Lambda-A*X/2)/beta;
Y=rooth(-Y_,eta/(2*beta)).^2;
Y = min(Y,1);
Y=(Y.^2*beta/2-beta*Y.*Y_+eta*Y.^0.5<0).*Y;

%fprintf('%i\t%f\t%f\n',iter,norm(X-Y,'fro'),obj(X,Y))
% saved_obj = [saved_obj obj(X,Y)];
%updata Lambda
Lambda = Lambda+beta*(X-Y);

% 终止准则
del_Y = temp_Y - Y;
r_matrix = abs(-0.5 .* A * del_Y + beta .* del_Y);
r = max(r_matrix(:));
XY = X-Y;
d_XY = max(abs( XY(:)  ));
error = max(r,d_XY) ;
fprintf('%i,%f,%f\n',iter,r,d_XY)
error_list = [error_list error ];
toc
if error < epsi
    break;
end
end
% 
% plot_x = [1: maxiter];
% plot(plot_x, saved_obj);
% xlabel('Iterations');
% ylabel('Objective Value'); 
    
    function fun = obj(X,Y)
        fun = sum(Y.*(A*X),'all')/2+sum(G.*X,'all')+sum(Lambda.*(X-Y),'all')+beta/2*norm(X-Y,"fro")^2+eta*sum(Y,'all');
    end
    
    function fun = ori_obj(X,Y)
        fun = sum(X.*(A*Y),'all')/2+sum(G.*Y,'all');
    end

end


%%
function y =  rooth(p,q)
% x^3+px+q=0 the maximal root
if q==0
    y=max(0,-p).^0.5;
else
delta=(q/2).^2+(p/3).^3;
d=(delta).^0.5;
a1=(-q/2+d).^(1/3);
a2=(-q/2-d).^(1/3);
w=(3^0.5*1i-1)/2;
y1=a1+a2;
y2=w*a1+w^2*a2;
y3=w^2*a1+w*a2;
% n1=norm(imag(y1));
y=max(real(y1),max(real(y2),real(y3))).*(delta<0);
% y=real(y);
end
end