function [hist,Lencliq] = orthbatch(x,Q,G,stepsize,bs,nepoch,Qa,thres,opt_type,gamma,lamb)

[n,m] = size(G);
ga = sum(G,2);
nb = m / bs;
hist = zeros(nepoch+1, 1); hist(1) = getobj(x,Qa,ga);
C = cell(1,nb);
Lencliq = zeros(nepoch, 1);
iter_count = 0;
for epoch = 1: nepoch
    if epoch == 1
        I = reshape(randperm(m),bs,nb);
    else
        [I,lencliq] = C2I(I,C,thres);
        perm = randperm(nb);
        I = I(:,perm);
        Lencliq(epoch) = mean(lencliq);
    end
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
            s = beta1 * s + (1 - beta1) * grad;
            v = beta1 * v + (1 - beta2) * grad .* grad;
            s2 = s / (1 - beta1^iter_count);
            v2 = v / (1 - beta2^iter_count);
            x = x - stepsize * s2 ./ (sqrt(v2) + epsilon);
        end
        Grad_normal = Grad .* repmat( sum(Grad.^2).^-0.5,n,1 );
        Ctemp = Grad_normal' * Grad_normal;
        C{iter} = Ctemp ; % - diag(diag(Ctemp));
       
    end
    hist(epoch+1) = getobj(x,Qa,ga);
     
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
function D = getclique(R)
    A = R~=0;
    n = size(R,2);
    C = 1:n; % candidate
    D = []; % determined
    while ~isempty(C)
    [maxdegree,i] = max(sum(A(C,C)));
    D = [D,C(i)];
    C = C(A(C(i),C));
    end
end

function [I , len] = C2I(I0,C,thres)
    I = zeros(size(I0));
    [ bs, nb ] = size(I0);
    len = zeros(1,nb);
    J = [];
    for i = 1:nb
%         inI0 = getclique((1-abs(C{i})).*(abs(C{i})<=thres));
        inI0 = getclique(C{i}<=thres);
        Itemp = I0( inI0 ,i);
        len(i) = length(Itemp);
        I(1:len(i), i) = Itemp;
        L = true(1,bs); L(inI0) = false; J = [J;I0( L ,i)];
    end
    perm = randperm(length(J));
    J = J(perm);
    ind = 0;
    for i = 1:nb
        I(len(i)+1:end, i) = J(ind+1:ind+bs-len(i));
        ind = ind +bs-len(i);
    end

end



