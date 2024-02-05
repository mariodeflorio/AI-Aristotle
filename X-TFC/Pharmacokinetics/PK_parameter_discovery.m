%%
clear; close all; clc;
% format long
%--------------------------------------------------------------------------
%{ 
  X-TFC applied to Systems Biology
  Test Case - Ultradian endocrine model

  Authors:
  Mario De Florio
%}
%%
%--------------------------------------------------------------------------
%% Input

rng('default') % set random seed

file_path = 'drug_real_10.csv';
data = readmatrix(file_path);

noise_lev = 0;

start = tic;

t_0 = 0; % initial time
t_f = 50; % final time [min]

N = 100;    % numer of collocation points of the NN
m = 100;    % number of neurons

t_step = 50;

x = linspace(0,1,N)'; % Discretization of collocation points

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

final_subdom = (n_t - 1); % consider all domain
%final_subdom = 1; % consider only first sub-domain

N_test = 100; % number of test points per each subdomain
x_test = linspace(0,1,N_test)'; % Discretization of collocation points

n_points = n_t + (n_t-1)*(N-2);
t_domain = linspace(t_0,t_f,n_points);

% iterative least-square parameters

IterMax = 100;
IterTol = 1e-6;

type_act = 2; % activation functions

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

%{
 
1= Logistic;
2= TanH;
3= Sine;
4= Cosine;
5= Gaussian; the best so far w/ m=11
6= ArcTan;
7= Hyperbolic Sine;
8= SoftPlus
9= Bent Identity;
10= Inverse Hyperbolic Sine
11= Softsign
 
%}

%% =======================================
% synthetic data (EXACT SOLUTION)

y1_anal = data(:,2);
y2_anal = data(:,3);
y3_anal = data(:,4);

t_obs = linspace(t_0,t_f,length(y1_anal));

%% interpolation observed data

y_RK_inter = 1:length(y1_anal);
ind = linspace(1,length(y1_anal),n_points);

y1_data = spline(y_RK_inter,y1_anal,ind)';
y2_data = spline(y_RK_inter,y2_anal,ind)';
y3_data = spline(y_RK_inter,y3_anal,ind)';


%% define activation functions 

weight = unifrnd(LB,UB,m,1);
bias = unifrnd(LB,UB,m,1);

h= zeros(N,m); hd= zeros(N,m); hdd= zeros(N,m);

for i = 1 : N
    for j = 1 : (m)
        [h(i, j), hd(i, j), hdd(i,j)] = act(x(i),weight(j), bias(j),type_act);
    end
end

h0 = h(1,:); 

% basis for test

h_test= zeros(N_test,m); hd_test= zeros(N_test,m); hdd_test= zeros(N_test,m);

for i = 1 : N_test
    for j = 1 : (m)
        [h_test(i, j), hd_test(i, j), hdd_test(i,j)] = act(x_test(i),weight(j), bias(j),type_act);
    end
end

h0_test = h_test(1,:); 

%% 
Z = zeros(N,m);
z = zeros(N,1);

% Initial Values

y1_initial = 0; 
y2_initial = 0.1; 
y3_initial = 0;

y1_0 = y1_initial;
y2_0 = y2_initial;
y3_0 = y3_initial;

sol1 = zeros(n_points,1); 
sol2 = zeros(n_points,1); 
sol3 = zeros(n_points,1); 

k_g_discover_vec = zeros(n_t-1,1); 
k_b_discover_vec = zeros(n_t-1,1); 

training_err_vec = zeros(n_t-1,1);

tStart = tic;

% parameters initial guess
k_g = 0;
k_b  = 0;

for i = 1:final_subdom 
     
    xi_1 = zeros(m,1);
    xi_2 = zeros(m,1);
    xi_3 = zeros(m,1);

    y1_data_i = y1_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y2_data_i = y2_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y3_data_i = y3_data((N-1)*(i-1)+1:(N-1)*i+1) ;

    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));

    t = linspace(t_tot(i),t_tot(i+1),N)' ;
    
    xi = [xi_1;xi_2;xi_3;k_g;k_b];

    %% Build Constrained Expressions
    
    y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;     
    y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;   
    y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
   
    %% Build the Losses  

    L_1 = - y1_dot  +  k_g*y2  -  k_b*y1 ;
    L_2 = - y2_dot  -  k_g*y2      ;
    L_3 = - y3_dot  +  k_b*y1;

    L_data_1 = y1_data_i - y1;
    L_data_2 = y2_data_i - y2;
    L_data_3 = y3_data_i - y3;

    Loss = [L_1 ; L_2 ; L_3; L_data_1 ; L_data_2 ; L_data_3];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;

    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);

        %% compute derivatives

        % L1
        L_y1_xi_1 = - c_i*hd  - k_b*(h-h0)    ;
        L_y1_xi_2 = k_g*(h-h0) ;
        L_y1_k_g = y2 ;
        L_y1_k_b = -y1 ;

        %L2
        L_y2_xi_2 = - c_i*hd - k_g*(h-h0) ;
        L_y2_k_g = - y2 ;

        %L3
        L_y3_xi_1 =  k_b*(h-h0)  ;
        L_y3_xi_3 = - c_i*hd ;
        L_y3_k_b = y1 ;

        %% Jacobian matrix     

        JJ = [ L_y1_xi_1 , L_y1_xi_2 ,     Z     ,  L_y1_k_g  ,  L_y1_k_b  ; 
                   Z     , L_y2_xi_2 ,     Z     ,  L_y2_k_g  ,      z     ;
               L_y3_xi_1 ,     Z     , L_y3_xi_3 ,      z     ,  L_y3_k_b  ;
                -(h-h0)  ,     Z     ,     Z     ,      z     ,      z     ;
                   Z     ,  -(h-h0)  ,     Z     ,      z     ,      z     ;
                   Z     ,     Z     ,  -(h-h0)  ,      z     ,      z     ];

            
        % xi variation
        
        dxi = lsqminnorm(JJ,Loss);
        

        % update xi
        xi = xi - dxi;
        
        xi_1 = xi((0*m)+1:1*m);
        xi_2 = xi((1*m)+1:2*m);
        xi_3 = xi((2*m)+1:3*m);      
        k_g = xi(3*m+1);
        k_b = xi(3*m+2);
        

        %% Re-Build Constrained Expressions
        
        y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;
        y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;
        y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
      
        
        %% Re-Build the Losses

        L_1 = - y1_dot  +  k_g*y2  -  k_b*y1 ;
        L_2 = - y2_dot  -  k_g*y2      ;
        L_3 = - y3_dot  +  k_b*y1;
        L_data_1 = y1_data_i - y1;
        L_data_2 = y2_data_i - y2;
        L_data_3 = y3_data_i - y3;

        Loss = [L_1 ; L_2 ; L_3 ; L_data_1 ; L_data_2 ; L_data_3];
        
        l2(2) = norm(Loss);
        iter = iter+1;
        
    end
     
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2)))   ;    
    % Update of constraints
    
    y1_0 = y1(end);
    y2_0 = y2(end);
    y3_0 = y3(end);
       
	sol1((N-1)*(i-1)+1:(N-1)*i+1) = y1;
    sol2((N-1)*(i-1)+1:(N-1)*i+1) = y2;
    sol3((N-1)*(i-1)+1:(N-1)*i+1) = y3;

    training_err_vec(i) = training_err;

    k_g_discover_vec(i) = k_g ;
    k_b_discover_vec(i) = k_b ;
    
   
end

xtfc_elapsedtime = toc(tStart) ;

fprintf('\n')
fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );


%% errors

if final_subdom == 1
    fprintf('\n')
    fprintf(' The value of the discovered parameter k_g is: %12.12f \n', k_g )
    fprintf(' The value of the discovered parameter k_b is: %12.12f \n', k_b )
    fprintf('\n')
    fprintf(' The relative error for k_g is: %.6g %%\n', 100*(abs(k_g - 0.72))/0.72 )
    fprintf(' The relative error for k_b is: %.6g %%\n', 100*(abs(k_b - 0.15)/0.15 ))
else
    fprintf('\n')
    fprintf(' The average value of the discovered parameter k_g is: %12.12f \n', mean(k_g_discover_vec) )
    fprintf(' The average value of the discovered parameter k_b is: %12.12f \n', mean(k_b_discover_vec) )
    fprintf('\n')
    fprintf(' The relative error for k_g is: %.6g %%\n', 100*(abs(mean(k_g_discover_vec) - 0.72))/0.72 )
    fprintf(' The relative error for k_b is: %.6g %%\n', 100*(abs(mean(k_b_discover_vec) - 0.15)/0.15 ))
end





