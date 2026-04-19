import torch
import numpy as np
import sympy as sp
import cmath
import time
import os
import pandas as pd
def rooth(p, q,device):
    # delta = (q / 2) ** 2 + (p / 3) ** 3
    # for 
    # t = sp.Symbol('t')
    # t_p = p.cpu().numpy()
    # t_q = q.cpu().numpy()
    # f = t **3 + t_p*t**2 + t_q
    # y = sp.solve(f)
    # y1,y2,y3 = y[0],y[1],y[2]
    # y = torch.max(y1.real, torch.max(y2.real, y3.real)) * (delta < 0).to(torch.float32)
    # return y.to(torch.float32)

    if q == 0:
        return torch.max(torch.zeros_like(p), -p).sqrt().to(torch.float32)
    else:
        delta = (q / 2) ** 2 + (p / 3) ** 3
        n,k = delta.shape
        delta_cpu = delta.detach().cpu()
        d_real = torch.ones_like(delta)
        d_imag = torch.ones_like(delta)
        delta_cpu = delta.cpu().numpy().astype(complex)
        d = delta_cpu ** (1/2)
        d = torch.tensor(d)
        d = d.to(device)
        # for i in range(n ):
        #     for j in range(k):
        #         t = cmath.sqrt(delta[i,j])
        #         d_real[i,j] = t.real
        #         d_imag[i,j] = t.imag
        # d_complex = torch.complex(d_real, d_imag)         
        # d_complex = torch.sqrt(delta)        
        # d = d_complex.to(device)
        # d = torch.tensor([[cmath.sqrt(complex(y.item(), 0)).real for y in row] for row in delta], dtype=torch.float).to(device)
        a1 = ((-q / 2 + d) ** (1 / 3))
        a2 = ((-q / 2 - d) ** (1 / 3))
        w = (3 ** 0.5 * 1j - 1) / 2
        y1 = a1 + a2
        y2 = w * a1 + w ** 2 * a2
        y3 = w ** 2 * a1 + w * a2
        y = torch.max(y1.real, torch.max(y2.real, y3.real)) * (delta < 0)
        del a1,a2,w,y1,y2,y3
        return y.to(torch.float32)
def Vector_method(A,n_p,  n,k,eta, beta, max_iter, X,Y,epsi,device, i,j ,output_path,test_each_iter_time,save_X_Y_dir):
    Lambda = torch.zeros(n, 1).to(torch.float32).to(device)
    A = A.to(torch.float32)
    en = torch.ones(n,1).to(device)
    error_list = []
    t_s = time.time()
    n = torch.tensor(n).float()
    k = torch.tensor(k).float()
    phase =1
    eta = eta + 0.01* np.random.rand()
    for iter in range(1, max_iter + 1):
        print(n_p,n)
        if test_each_iter_time:
            t11 = time.time()
        AY = A@Y
        #del beta_scale, eta_scale, obj_value
        # Lambda=Lambda.to(torch.float32)
        B = Y - (Lambda + AY / 2 ) / beta
        eB = torch.sum(B, dim=0)
        eeX = en * eB
        # X = X - 1 / n * en@en.T@X + k/n * en
        X = B - 1/n * eeX + k/n * en
        
        
        y_max = torch.max(Y)
        temp_Y = Y
        Y_ = X + (Lambda - A @ X / 2) / beta
        Y = rooth(-Y_, eta / (2 * beta), device) ** 2
        Y = torch.min(Y, torch.ones_like(Y))
        Y = torch.where((Y ** 2 * beta / 2 - beta * Y * Y_ + eta * Y.sqrt() < 0), Y, torch.zeros_like(Y))
        # Y[Y > 0.995] = 1
        # Y[Y<0.005] = 0
        Lambda = Lambda + beta * (X - Y)
        del_Y = Y - temp_Y
        Y_ = torch.abs(-0.5 * A @ del_Y + beta * del_Y)
        r = torch.max(Y_)
        XY = X - Y
        d_XY = torch.max(torch.abs(XY))
        error = max(r.item(), d_XY.item())
        error_list.append(error)
       
        beta_scale = beta * torch.norm(XY, 'fro') ** 2
        eta_scale = eta* torch.sum(Y)
        obj_value = X.T @ AY / 2  + torch.sum(Lambda.T @ (X - Y))
        print(f"Iteration: {iter}, Beta Scale: {beta_scale}, Eta Scale: {eta_scale.item()}, Objective Value: {obj_value.item()}")
       
       
        print(i,iter, r.item(), d_XY.item(),y_max)
        nonzero_indices = torch.nonzero(Y)
        nnz_Y = nonzero_indices.size(0)
        output_string = "batch:{},outer_iter:{},inner_iter:{},r:{:.2g},nnz_Y:{},beta:{:.2g},eta:{:.2g}\n".format(i,j,iter,r.item(), nnz_Y, beta, eta)
        print(output_string)
        if test_each_iter_time:
            t22 = time.time()
            t_one = t22 - t11
            print(f"Each_iter_time: {t_one:.2f} seconds\n")
        if iter % 1000 ==0:
            t_e = time.time()
            # t_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_s))
            # t_e = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_e))
            t_u = t_e - t_s
            with open(output_path, 'a') as f:
                f.write(f"Batch:{i}Iteration {j}: iter={iter}, r={r.item()}, d_XY={d_XY.item()}, y_max={y_max}\n")
                f.write(f"Total time: {t_u:.2f} seconds\n")
        if error < epsi:
            break
    
    return X, Y, error_list
    


def Matirx_method(A, G, eta, beta, max_iter, X,Y,Lambda,epsi,device, i ,output_path ,test_each_iter_time,save_X_Y_dir):
    n, k = G.size()
    
    
    en = torch.ones(n,1).to(device)
    ek = torch.ones(k,1).to(device)
    error_list = []
    t_s = time.time()
    obj0 = float("inf")
    ny0 = n*k
    count =0
    phase =1
    save_2000 =0
    save_20 =0
    save_10 = 0
    save_5 =0
    Y = Y.to(torch.float32)
    Lambda=Lambda.to(torch.float32)
    AY_new = A @ Y
    for iter in range(1, max_iter + 1):
        if test_each_iter_time:
            t11 = time.time()
        #beta_scale = beta * torch.norm(X - Y, 'fro') ** 2
        #eta_scale = eta* torch.sum(Y)
        #obj_value = torch.sum(Y * (A @ X)) / 2 + torch.sum(G * X) + torch.sum(Lambda * (X - Y))
        #print(f"Iteration: {iter}, Beta Scale: {beta_scale}, Eta Scale: {eta_scale.item()}, Objective Value: {obj_value.item()}")
        #del beta_scale, eta_scale, obj_value
        temp_X = X
        AY_old  = AY_new 
        X = Y - (Lambda + AY_old / 2 + G) / beta
        eX = torch.sum(X, dim=0)
        eeX = eX.repeat(n, 1)
        Xe = torch.sum(X, dim=1)
        X_p2 = 1 / k * (en - Xe.unsqueeze(1) + 1 / n * torch.sum(eeX, dim=1).unsqueeze(1))
        X = X - 1 / n * eeX + X_p2.repeat(1,k)
        # X[X > 0.95] = 1
        # X[X<0.05] = 0
        del eX, eeX, Xe, X_p2
        # X = X-1/n*en*en'*X+1/k*(en-X*ek+  1/n* en *en'*X *ek)*ek'
        # p1_X = (1/n) * torch.matmul(en, torch.matmul(en.transpose(0, 1), X))
        # P21_X = en - torch.matmul(X, ek) + (1/n) * torch.matmul(en,torch.matmul( torch.matmul(en.transpose(0, 1), X),ek) )
        # P2_X = (1/k) * torch.matmul(P21_X, ek.transpose(0, 1))
        # X = X - p1_X + P2_X
        y_max = torch.max(Y)
        temp_Y = Y
        Y_ = X + (Lambda - A @ X / 2) / beta
        Y = rooth(-Y_, eta / (2 * beta), device) ** 2
        Y = torch.min(Y, torch.ones_like(Y))
        Y = torch.where((Y ** 2 * beta / 2 - beta * Y * Y_ + eta * Y.sqrt() < 0), Y, torch.zeros_like(Y))
        
        # Y[Y > 0.95] = 1
        # Y[Y<0.005] = 0
        Lambda = Lambda + beta * (X - Y)
        del_Y = Y - temp_Y
        AY_new = A @ Y 
        Y_ = torch.abs(-0.5 *(AY_old - AY_new) + beta * del_Y)
        r = torch.max(Y_)
        #r = max( abs( torch.sum(Y == 1, dim=0) - (n/k) ) )
        XY = X - Y
        d_XY = torch.max(torch.abs(XY))
        dy = torch.norm(del_Y, 'fro')
        dxy = torch.norm(XY,'fro')
        error = max(r.item(), d_XY.item())
        error_list.append(error)
        print(i,iter, r.item(), d_XY.item(),y_max)
        # obj1 = fuzhu_obj( X, temp_X, Y, A, G, Lambda, beta, eta     )
        obj1 = obj(X,Y,A, G, Lambda, beta, eta )

        nonzero_indices = torch.nonzero(Y)
        nnz_Y = nonzero_indices.size(0)
        output_string = "outer_iter:{},inner_iter:{},phase:{},r:{:.2g},nnz_Y:{},beta:{:.2g},eta:{:.2g},dxy:{:.2g},dy{:.2g},obj:{:.6g}\n".format(i,iter, phase, r.item(), nnz_Y, beta, eta, dxy.item(), dy.item(), obj1.item())
        print(output_string)
        # if ny0 == n:
        #     eta = eta /2
        # if (iter%100) == 0:   # 更新参数
        if iter > 10000000 == 0:   # 更新参数    
            # switch 语句的实现
            if phase == 1:  # unstable stage
                if obj1 < obj0:
                    count = count + 1
                    if count == 5:
                        phase = 2
                        count = 0
                else:
                    beta = beta * 2
                    count = 0
            elif phase == 2:  # reduce beta large step
                if obj1 < obj0 and dxy < dy :
                    beta = beta / 2
                else:
                    beta = beta * 2
                    phase = 3
            elif phase == 3:  # reduce beta small step
                if obj1 < obj0 and dxy < dy :
                    beta = beta / 1.1
                else:
                    beta = beta * 1.1
                    phase = 4
            elif phase == 4:  # stable stage
                if obj1 < obj0 * 0.95:
                    if iter % 10 == 0:
                        ny1 = torch.nonzero(Y).size(0)
                        if ny0 != n:
                            ratio = (ny0 - ny1) / (ny0 - n)
                        else:
                            ratio = 0
                        if ratio >0 and ratio < 2e-2 and eta / beta < 0.005 :
                            beta = beta * 1.1
                            eta = eta * 2
                            # eta = torch.clamp(eta1, max=0.001)
                        ny0 = ny1
                else:
                    phase = 1
                    beta = beta * 2

        # obj0 = fuzhu_obj( X, temp_X, Y, A, G, Lambda, beta, eta     )
        # obj0 = obj(X,Y, A, G, Lambda, beta, eta     )
        
        
        
        if test_each_iter_time:
            t22 = time.time()
            t_one = t22 - t11
            print(f"Each_iter_time: {t_one:.2f} seconds\n")
        if iter % 1 ==0:
            t_e = time.time()
            # t_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_s))
            # t_e = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_e))
            t_u = t_e - t_s
            with open(output_path, 'a') as f:
                f.write(f"Iteration {i}: iter={iter}, r={r.item()},beta:{beta},eta:{eta}, d_XY={d_XY.item()}, y_max={y_max}\n")
                f.write(f"Total time: {t_u:.2f} seconds\n")
                f.write(output_string + '\n')
                # torch.save(Y, f"{save_X_Y_dir}_Y_{iter}.pt")
                # t_Y =Y.cpu().numpy()
                # t_X =X.cpu().numpy()
                # t_lambda = Lambda.cpu().numpy()
                # np.savetxt(f"{save_X_Y_dir}_Y_{iter}.csv", t_Y, delimiter=",")
                # np.savetxt(f"{save_X_Y_dir}_X_{iter}.csv", t_X, delimiter=",")
                # np.savetxt(f"{save_X_Y_dir}_LAMBDA_{iter}.csv", t_lambda, delimiter=",")
        non_zero_one_indices = torch.nonzero((Y != 0) & (Y != 1))
        non_zero_one_count = non_zero_one_indices.size(0) 
        one_indices = torch.nonzero((Y ==1))
        one_indices_count = one_indices.size(0)
        if (iter % 5000) == 0:
            t_Y =Y.cpu().numpy()
            t_X =X.cpu().numpy()
            t_lambda = Lambda.cpu().numpy()
            np.savetxt(f"{save_X_Y_dir}_Y_nnzY_{nnz_Y}_iter_{iter}.csv", t_Y, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_X_nnzY_{nnz_Y}_iter_{iter}.csv", t_X, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_LAMBDA_nnzY_{nnz_Y}_iter_{iter}.csv", t_lambda, delimiter=",")
        if non_zero_one_count < 4000 and one_indices_count > 59880 and save_2000 == 0:
            save_2000 = 1
            with open(output_path, 'a') as f:
                f.write(f"Iteration {i}: iter={iter}, r={r.item()},beta:{beta},eta:{eta}, d_XY={d_XY.item()}, y_max={y_max},less than 20!!!\n")
            t_Y =Y.cpu().numpy()
            t_X =X.cpu().numpy()
            t_lambda = Lambda.cpu().numpy()
            np.savetxt(f"{save_X_Y_dir}_Y_less4000_{iter}.csv", t_Y, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_X_less4000_{iter}.csv", t_X, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_LAMBDA_less4000_{iter}.csv", t_lambda, delimiter=",")

        if non_zero_one_count < 20 and one_indices_count > 59880 and save_20 == 0:
            save_20 = 1
            with open(output_path, 'a') as f:
                f.write(f"Iteration {i}: iter={iter}, r={r.item()},beta:{beta},eta:{eta}, d_XY={d_XY.item()}, y_max={y_max},less than 20!!!\n")
            t_Y =Y.cpu().numpy()
            t_X =X.cpu().numpy()
            t_lambda = Lambda.cpu().numpy()
            np.savetxt(f"{save_X_Y_dir}_Y_less20_{iter}.csv", t_Y, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_X_less20_{iter}.csv", t_X, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_LAMBDA_less20_{iter}.csv", t_lambda, delimiter=",")
        if non_zero_one_count < 10 and one_indices_count > 59884 and save_10 == 0:
            save_10 = 1
            with open(output_path, 'a') as f:
                f.write(f"Iteration {i}: iter={iter}, r={r.item()},beta:{beta},eta:{eta}, d_XY={d_XY.item()}, y_max={y_max},less than 10!!!\n")
            t_Y =Y.cpu().numpy()
            t_X =X.cpu().numpy()
            t_lambda = Lambda.cpu().numpy()
            np.savetxt(f"{save_X_Y_dir}_Y_less10_{iter}.csv", t_Y, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_X_less10_{iter}.csv", t_X, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_LAMBDA_less10_{iter}.csv", t_lambda, delimiter=",")
        if non_zero_one_count < 5 and one_indices_count > 59900 and save_5 == 0:
            save_5 = 1
            with open(output_path, 'a') as f:
                f.write(f"Iteration {i}: iter={iter}, r={r.item()},beta:{beta},eta:{eta}, d_XY={d_XY.item()}, y_max={y_max},less than 5!!!\n")
            t_Y =Y.cpu().numpy()
            t_X =X.cpu().numpy()
            t_lambda = Lambda.cpu().numpy()
            np.savetxt(f"{save_X_Y_dir}_Y_less5_{iter}.csv", t_Y, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_X_less5_{iter}.csv", t_X, delimiter=",")
            np.savetxt(f"{save_X_Y_dir}_LAMBDA_less5_{iter}.csv", t_lambda, delimiter=",")
        
        
            
        if error < epsi and r.item() ==0 :
            break
    
    return X, Y, error_list




    
    
def obj(X, Y, A, G, Lambda, beta, eta):
    return torch.sum(Y * (A @ X)) / 2 + torch.sum(G * X) + torch.sum(Lambda * (X - Y)) + beta / 2 * torch.norm(X - Y, 'fro') ** 2 + eta * torch.sum(Y)
def fuzhu_obj (X1,X0,Y,A,G,Lambda,beta,eta):
    n,k = Y.size()
    return torch.sum(Y * (A @ X1)) / 2 + torch.sum(G * X1) + torch.sum(Lambda * (X1 - Y)) + beta / 2 * torch.norm(X1 - Y, 'fro') ** 2 + eta * torch.sum(Y) + 2* n**2 * k**2 / beta * torch.norm(X1-X0, "fro")




# # Example usage
# A = torch.rand(60000, 60000).float()
# G = torch.rand(60000, 49).float()
# eta = torch.tensor(1.0,dtype=torch.float32)
# eta = eta.to(torch.float32)
# beta = torch.tensor(100.0,dtype=torch.float32)
# max_iter = 100
# in_X = torch.rand(60000, 49).float()
# in_Y = torch.rand(60000, 49).float()
# epsi = 1e-4
# X, Y, error_list = splitadmm20240119(A, G, eta, beta, max_iter, in_X, in_Y,epsi)
