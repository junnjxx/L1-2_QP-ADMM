% function [X,Y,error_list,flag,Lambda_history,r_collect,dxy_collect,round_y_collect,y_non_ratio,lambda_fnorm,y_history,x_history,R_history] = splitadmm20240119(A,G,eta,beta,i,iter, X,Y,epsi)
function [X,Y,error_list,flag,Lambda_history,r_collect,dxy_collect,round_y_collect,y_non_ratio,lambda_fnorm] = splitadmm20240119(A,G,eta,beta,i,iter, X,Y,epsi)
    %40  r_up r_lo 0.01 0.005  100
    r_up = 0.01; r_lo = 0.005;ratio_step = 100; % 80
     % r_up = 0.005; r_lo = 0.0001;ratio_step = 100;
    % r_up = 0.01; r_lo = 0.005; ratio_step =100;
    %  ratio_step =50;
    % ratio_step = 1;
    flag = 0;
    [n,k] = size(G);
    en = ones(n,1);
    ek = ones(k,1);
    % Lambda = rand(n,k); % 看看lambda随机取0-1
    Lambda = zeros(n,k);
    maxiter = iter;
    iter = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            j
    ny0 = n*k;
    count = 0;
    phase = 1;
    obj0 = inf ;
    %fprintf('%i\t%f\t%f\n',iter,norm(X-Y,'fro'),obj(X,Y))
    error_list = [];
    Lambda_history = cell(maxiter, 1);  % 用于记录每次迭代的 Lambda
    saved_obj = [];
    r_collect = [];
    dxy_collect = [];
    round_y_collect = [];
    y_non_ratio = [];
    lambda_fnorm =[];
    % y_history = zeros(k*n, maxiter); 
    % R_history = zeros(k*n, maxiter); 
    % x_history = zeros(k*n, maxiter); 
    % lambda_history = zeros(k*n, maxiter);
    if mod(iter,ratio_step) == 0
    fprintf('n_iter: %d,beta: %f,eta:%f\n',i,beta,eta)
    end
    AY_new = A*Y;
    dXY0 = norm(Y-X);
    for iter = 1:maxiter

        %update X
        old_X =X;
        AY = AY_new;
        X = Y-(Lambda+AY/2+G)/beta;
        eX = sum(X,1);
        X = X-eX/n+(1+sum(eX)/n-sum(X,2))/k;
        temp_Y = Y;
        XY = X -Y;
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
        % y_history(:, iter)  = Y(:);
        % R_history(:, iter)  = Y_(:);
        % x_history(:, iter)  = X(:);
        % if iter < 1000
        % Y(Y>0.95) =1;
        % end 
        % Y_ = X+(Lambda-A*X/2)/beta;
        % Y=rooth(-Y_,eta/(2*beta)).^2;
        % Y = min(Y,1);
        % Y=(Y.^2*beta/2-beta*Y.*Y_+eta*Y.^0.5<0).*Y;
        % AY_new = A * Y;
        %updata Lambda
        dXY = norm(Y-X,'fro');
        if dXY ~= 0 
            % gamma =  min(1, log2(2)^2* dXY0/(iter*(log2(iter+1))^2*dXY));
            gamma =  min(1, 10000*log2(2)^2 * dXY0 / (iter * (log2(iter))^2 *dXY  )  )   ;
            fprintf ('gamma:%i \n',gamma)
            beta_ =beta * gamma;
            beta_ =beta_ ;
        else
            beta_ = beta;
        end 
        Lambda = Lambda+beta_*(X-Y);
        % lambda_history(:, iter) =  Lambda(:);
        % Lambda_history{iter} = Lambda;
        % obj1 = obj(old_X,temp_Y,temp_lamda,beta,eta) - 2*n*n*k*k/beta * norm(old_X-X,"fro");
        obj1 =  obj(X,Y,Lambda,beta,eta);
        % 终止准则
        interval = 1;
        if mod(iter,interval) ==0  %每隔interval次算一下终止准则/更新beta eta
            del_Y = temp_Y - Y;
            % XY = X-Y;  % 这里是  X^k+1 - Y^k+1 
            d_XY = max(abs( XY(:)  ));
            dy = norm(del_Y,'fro');
            dxy = norm(XY,'fro');
            
            r_matrix = 0.5 .* AY_new - 0.5 .* AY + beta .* del_Y;
            r = norm(r_matrix,'fro');
            error = max(r,dxy) ;

            % % 最近的0/1整数：小于0.5的取0，大于等于0.5的取1
            % Y_binary = round(Y);  % 四舍五入到0或1
            % element_deviation = abs(Y - Y_binary);  % 每个元素的偏差
            % round_y_collect(iter) = norm(element_deviation,'fro');  % 总偏差（F2范数）
            % % 统计非0/1的元素数量（二进制误差）
            % y_non_ratio(iter) = nnz(element_deviation)/(n*k);
            % 
            % if mod(iter,ratio_step) == 0
            % fprintf('iter:%i,r:%f,d_XY:%f\n',iter,r,d_XY)
            % end
            % 
            % lambda_fnorm(iter) = norm(Lambda,'fro');
            % r_collect(iter) = r;
            % dxy_collect(iter) = dxy;
            % 
            % 测试有限终止性质的 % 结果发现不行 dxy不可能变成0
            % if error == 0  
            %     flag =1;
            %     break;
            % end



            if error < epsi %|| iter > 1 % 当eta很小很小的时候，收敛不到r=0的位置&& r == 0
                flag =1;
                break;
            end
            
            if mod(iter,ratio_step) == 0
            fprintf('iter:%i, phase: %i,nnzY: %i,beta:%4.2g,eta:%4.2g,dxy:%4.2g,dy:%4.2g\n',iter,phase,nnz(Y),beta_,eta,dxy,dy)  
            end
            obj_part = sum(Y.*(A*X),'all')/2+sum(G.*X,'all')+sum(Lambda.*(X-Y),'all')  ;
            beta_part = beta/2*norm(X-Y,"fro")^2;
            eta_part = eta*sum(Y.^(1/2),'all');
            if mod(iter,ratio_step) == 0
            fprintf('iter:%i, phase: %i, obj_part:%4.2g, beta_part:%4.2g, eta_part:%4.2g, total_obj:%4.2g\n',iter,phase,obj_part,beta_part,eta_part,obj_part+beta_part+eta_part);
            end
            if nnz(Y) < n*1.05
                phase = 6;
            end


            if mod(iter,1) ==0   &&   iter > 1 %% 自适应调节beta eta的条件
                switch phase
                    case 1 % unstable stage
                        if obj1 < 0
                            if obj1 < obj0 * 0.995
                                reduce = 1;
                            end
                        else
                            if obj1 < obj0 * 1.005
                                reduce = 1;
                            end
                        end
                    if reduce == 1
                        count = count+1;
                        if count == 10
                            phase=2;count=0;
                        end
                    else
                        beta = beta*2;
                        obj1 = obj(X,Y,Lambda,beta,eta);
                        count = 0;
                    end
                    case 2 % reduce beta large step
                    if obj1<obj0 && dxy < 0.1*dy
                        beta = beta/2;
                        obj1 = obj(X,Y,Lambda,beta,eta);
                    else
                        beta = beta*2;
                        phase = 3;
                        obj1 = obj(X,Y,Lambda,beta,eta);
                    end
                    case 3 % reduce beta small step
                            if obj1<obj0 && dxy<0.1*dy
                                beta = beta/1.1;
                                %eta = eta * 2; %%%
                                obj1 = obj(X,Y,Lambda,beta,eta);
                            else
                                beta = beta*1.1;
                                phase = 4;
                                ny0 = nnz(Y);
                                obj1 = obj(X,Y,Lambda,beta,eta);
                            end
                    case 4 
                        if mod(iter,ratio_step) == 0
                            ny1 = nnz(Y);
                            ratio = abs(ny1 - ny0) / (n*k);
                            fprintf('nnz0:%d, nnz1:%d, ratio:%f',ny0,ny1,ratio);
                            if ratio == 0
                                beta = beta/ 1.1;
                                eta = eta *1.1;
                            end
                            if ratio >= r_up%0.02
                                beta = beta * 1.2;
                                obj1 = obj(X,Y,Lambda,beta,eta);
                            elseif ratio < r_lo
                                beta = beta / 1.2;
                               
                                obj1 = obj(X,Y,Lambda,beta,eta);
                            else 
                                phase = 5;
                                ny0_c5 = nnz(Y);
                            end
                            ny0 = ny1 ;
                        end
                    case 5 % stable stage % 主要调节eta
                        if obj1 < 0
                            if obj1 < obj0 * 0.995
                                reduce = 1;
                            end
                        else
                            if obj1 < obj0 * 1.005
                                reduce = 1;
                            end
                        end
                        if reduce == 1
                            if mod(iter,ratio_step)==0
                                ny1_c5 = nnz(Y);                             
                                ratio  = abs( (ny0_c5-ny1_c5)/(n*k) );
                                fprintf('nnz0:%d, nnz1:%d, ratio:%f',ny1_c5,ny1_c5,ratio);
                                if ratio >= r_up%0.02
                                    phase = 4;
                                    beta = beta*1.2;
                                    eta = eta*1.2;
                                elseif ratio < r_lo
                                    phase = 4;
                                    beta = beta/1.2;
                                    eta = eta*1.2;
                                else
                                    % 这两个不确定
                                    % beta = beta*1.2;
                                    % eta = eta *  1.2 ; 
                                    obj1 = obj(X,Y,Lambda,beta,eta);
                                end

                               
                            ny0_c5 = ny1_c5;    
                            end
                        else
                            phase = 1;
                            beta = beta*2;
                            obj1 = obj(X,Y,Lambda,beta,eta);
                        end
                    case 6 
                        % if mod(iter,100) ==0
                        %     beta = beta * 1.1;
                        %     eta = eta/1.1;
                        % end

                  
                end



            end
            obj0 = obj1;

        end





        % if mod(iter, 100) == 0
        %     fid = fopen('config.txt','a');
        %     fprintf('%i,%f,%f\n',iter,r,d_XY)
        %     fclose(fid);
        % end

    
    end
    % y_history = y_history(:, 1:iter);
    % R_history = R_history(:, 1:iter);
    % x_history = x_history(:, 1:iter);
    % lambda_history = lambda_history(:,1:iter);
    % ay = lambda_history - (R_history-x_history)*beta;
% 
% plot_x = [1: maxiter];
% plot(plot_x, saved_obj);
% xlabel('Iterations');
% ylabel('Objective Value'); 
    
    function fun = obj(X,Y,Lambda,beta,eta)
        fun = sum(Y.*(A*X),'all')/2+sum(G.*X,'all')+sum(Lambda.*(X-Y),'all')+beta/2*norm(X-Y,"fro")^2+eta*sum(Y.^(1/2),'all')  ;
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