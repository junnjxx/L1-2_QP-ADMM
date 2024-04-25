function [X,Y,error_list,flag] = splitadmm20240119(A,G,eta,beta,n_iter,iter,X1,Y1,epsi)
flag = 0;
[n,k] = size(G);
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
fprintf('n_iter: %d,beta: %f,eta:%f\n',n_iter,beta,eta)
AY_new = A*Y;
for iter = 1:maxiter
tic;
if mod(iter, 100) == 0
    beta_scale = beta * norm(X-Y,"fro")^2;
    eta_scale = eta * sum(Y,'all');
    obj_value = sum(Y.*(A*X),'all')/2+sum(G.*X,'all')+sum(Lambda.*(X-Y),'all');
    fid = fopen('config.txt','a');
    fprintf('n_ iter:%i,iter:%i,beta_scale:%e,eta_scale:%e,obj_value:%e\n',n_iter,iter,beta_scale,eta_scale,obj_value)
    fclose(fid);
end
%update X
AY = AY_new;
X = Y-(Lambda+AY/2+G)/beta;
eX = sum(X,1);
X = X-eX/n+(1+sum(eX)/n-sum(X,2))/k;

temp_Y = Y;

%update Y
Y_ = X+(Lambda-A*X/2)/beta;
lamb = 2*eta/beta;
Yindex = Y_>3/4*lamb^(2/3);
y_ = Y_(Yindex);
y = 2/3*y_.*(1+cos(2/3*pi-2/3*acos((lamb)/8*(y_/3).^(-1.5))));
y((y-y_).^2+lamb*sqrt(y)-y_.^2>0) = 0;
y(y>1) = 1;
Y = zeros(n,k); 
Y(Yindex) = y;
AY_new = A * Y;


%updata Lambda
Lambda = Lambda+beta*(X-Y);

% 终止准则
del_Y = temp_Y - Y;
r_matrix = abs(0.5 .* AY_new - 0.5 .* AY + beta .* del_Y);
r = max(r_matrix(:));
XY = X-Y;
d_XY = max(abs( XY(:)  ));
error = max(r,d_XY) ;
if mod(iter, 100) == 0
    fid = fopen('config.txt','a');
    fprintf('%i,%f,%f\n',iter,r,d_XY)
    fclose(fid);
end
fprintf('%i,%f,%f\n',iter,r,d_XY)
error_list = [error_list error ];
if mod(iter,1000) == 0
    folder_path = './save/';
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end
    filename = sprintf('%ssaved_Y_%d_%d.csv', folder_path, n_iter, iter);
    csvwrite(filename, Y);
end
toc
if error < epsi
    flag =1;
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