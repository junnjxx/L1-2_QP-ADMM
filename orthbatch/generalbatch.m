function [hist,hist_x] = generalbatch(x,Q,G,stepsize,bs,nepoch,Qa,data_vector, input_I, opt_type,gamma,lamb,xstar)
[n,m] = size(G);
ga = sum(G,2);
nb = m / bs;
hist = zeros(nepoch+1, 1); hist(1) = getobj(x,Qa,ga);
hist_x = zeros(nepoch+1, 1); hist_x(1) = norm(x-xstar);
I  = input_I;
iter_count = 0;
for epoch = 1: nepoch
    random_order = randperm(size(I, 2));
    I = I(:, random_order);
    for iter =1: nb
        iter_count = iter_count +1;
        i = I(:,iter);
        Grad = getgrad(x,Q(i),G(:,i),opt_type,lamb,n);
        if opt_type == 0
            grad = sum(Grad,2);
            x = x - stepsize * grad;
        end
        if opt_type == 1
            grad = sum(Grad,2);
            x = x - stepsize * grad;
        end
        v = 0 ; 
        if opt_type == 2
            grad = sum(Grad,2);
            v = gamma .* v + stepsize .* grad;
            x = x - v;
        end
        s = 0 ; v = 0; beta1 = 0.9; beta2 = 0.999 ;epsilon = 1e-6;
        if opt_type == 3
            grad = sum(Grad,2);
            v = beta1 * v + (1 - beta1) * grad;
            s = beta2 * s + (1 - beta2) * grad .* grad;
            s2 = s / (1 - beta2^iter_count);
            v2 = v / (1 - beta1^iter_count);
            x = x - stepsize * v2 ./ (sqrt(s2) + epsilon);
        end
    end
    hist(epoch+1) = getobj(x,Qa,ga);
    hist_x(epoch+1) = norm(x-xstar); 
end
end
%%
function Grad = getgrad(x,Q,G,grad_type,lambda,n)
[n,m] = size(G);
Grad = zeros(n,m);
for i =1:m
    Grad(:,i) = Q{i}*x;
end
Grad = Grad + G;
if grad_type ==1
    Grad = Grad + (lambda/n .* x);
end


end
function f = getobj(x,Q,g)
    f = x'*Q*x/2+g'*x;
end







