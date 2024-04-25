function [x,y,saved_obj,error_list]  = splitadmm20240307(A,n, k, eta, beta, maxiter,X1,Y1)
% min 1/2*x'*A*x s.t. e'*x=k xi=0or1
% min 1/2*x'*A*y+eta*sum(y.^0.5) s.t. e'*x=k 0<=y<=1 x-y=0
%eta = norm(A,"inf"); %% todo
%eta = 0;  %%
x = X1;
e = ones(n,1);
y = Y1;
lambda = zeros(n,1);
epsi = 1e-5;
iter = 0;
%fprintf('%i\t%f\t%f\n',iter,norm(x-y),obj(x,y))
saved_obj = [];
error_list = [];

for iter = 1:maxiter


%update X
x = y-(lambda+A*y/2)/beta;
x = x-1/n*e*e'*x+k/n*e;

% fprintf('%i\t%f\t%f\n',iter,norm(x-y,'fro'),obj(x,y))
temp_y = y;
%update Y
y_ = x+(lambda-A*x/2)/beta;
q = eta/(2*beta);
% min 1/2*(y-y_).^2+eta*y.^0.5
y = rooth(-y_,q).^2;
% if y<=0, yopt=0
% if y>=1 yopt=0or1
% if 0<y<1 yopt=0ory
y = min(y,1);
y=(y.^2*beta/2-beta*y.*y_+eta*y.^0.5<0).*y;

%fprintf('%i\t%f\t%f\n',iter,norm(x-y,'fro'),obj(x,y))
saved_obj = [saved_obj obj(x,y)];
%updata Lambda
lambda = lambda+beta*(x-y);
r_matrix = abs(0.5 .* A * y - 0.5 .* A * temp_y + beta .* (temp_y - y));
s= max( abs(x-y));
r = max(r_matrix(:));
error = max(r,s);
fprintf('%i,%f,%f\n',iter,r,s)
error_list = [error_list error ];
if error < epsi
    break;
end



end


    function fun = obj(x,y)
        fun = 1/2*x'*A*y+eta*sum(y.^0.5);
    end


end


%%
function y =  rooth(p,q)
% x^3+px+q=0 the maximal root
%fprintf('%f,%f \n',p,q)
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