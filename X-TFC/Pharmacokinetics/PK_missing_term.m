%%
clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  X-TFC applied to Systems Biology
  Test Case - Ultradian endocrine model

  Author:
  Mario De Florio
%}
%%
%--------------------------------------------------------------------------
%% Input


rng('default') % set random seed

file_path = 'drug_real_100.csv';
data = readmatrix(file_path);

% Define the standard deviation of the Gaussian noise
noise_lev = 0.0; % Adjust this as needed [ 0 , 0.01 , 0.02 , 0.03 , 0.04 , 0.05 , 0.1]

if noise_lev == 0
    N = 26;    % Number of collocation points in each subdomain
    m = 100;    % number of neurons
    t_step = 12.5; % length of each subdomain
else
    N = 100;    
    m = 100;    
    t_step = 50;
end

IterTol = 1e-6;


start = tic;

t_0 = 0; % initial time
t_f = 50; % final time [min]

x = linspace(0,1,N)'; % Discretization of collocation points

N_test = 100; % number of test points per each subdomain
x_test = linspace(0,1,N_test)'; % Discretization of collocation points

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

n_points = n_t + (n_t-1)*(N-2);
t_domain = linspace(t_0,t_f,n_points);

% iterative least-square parameters

IterMax = 100;

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

%% Parameters definition

k_g = 0.72;
k_b  = 0.15;

%% =======================================
% synthetic data (EXACT SOLUTION)

y1_anal = data(:,2);
y2_anal = data(:,3);
y3_anal = data(:,4);
rhs_anal = data(:,5);

t_obs = linspace(t_0,t_f,length(y1_anal));

%% Data Perturbation

y1_data_pert = y1_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y1_anal) , 1));
y2_data_pert = y2_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y2_anal) , 1));
y3_data_pert = y3_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y3_anal) , 1));

% figure(1)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% plot(t_obs,y1_data_pert,'*','LineWidth',2)
% plot(t_obs,y2_data_pert,'*','LineWidth',2)
% plot(t_obs,y3_data_pert,'*','LineWidth',2)
% plot(t_obs,y1_anal,'LineWidth',2)
% plot(t_obs,y2_anal,'LineWidth',2)
% plot(t_obs,y3_anal,'LineWidth',2)
% ylabel('Tetracycline (mg)')
% xlabel('time (hours)')
% legend('GI tract (exact)', 'Bloodstream (exact)', 'Urinary tract (exact)', 'GI tract (inferred)', 'Bloodstream (inferred)', 'Urinary tract (inferred)')
% box on

%% interpolation observed data

y_RK_inter = 1:length(y1_data_pert);
ind = linspace(1,length(y1_data_pert),n_points);

y1_data = interp1(y_RK_inter,y1_data_pert,ind)';
y2_data = interp1(y_RK_inter,y2_data_pert,ind)';
y3_data = interp1(y_RK_inter,y3_data_pert,ind)';


y_RK_inter = 1:length(rhs_anal);
ind = linspace(1,length(rhs_anal),n_points);
rhs_data = interp1(y_RK_inter,rhs_anal,ind)';


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





%% Ax=b construction
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
rhs_vec = zeros(n_points,1); 

training_err_vec = zeros(n_t-1,1);

xi_1_vec = [];
xi_2_vec = [];
xi_3_vec = [];
xi_rhs_vec = [];

tStart = tic;

for i = 1:(n_t-1)

    xi_1 = zeros(m,1);
    xi_2 = zeros(m,1);
    xi_3 = zeros(m,1);
    xi_rhs = zeros(m,1);

    y1_data_i = y1_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y2_data_i = y2_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y3_data_i = y3_data((N-1)*(i-1)+1:(N-1)*i+1) ;

    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));

    t = linspace(t_tot(i),t_tot(i+1),N)' ;
    
    xi = [xi_1;xi_2;xi_3;xi_rhs];

    %% Build Constrained Expressions
    
    y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;     
    y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;   
    y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
    rhs = h*xi_rhs;
   
    %% Build the Losses  

    L_1 = - y1_dot  +  rhs ;
    L_2 = - y2_dot  -  k_g*y2      ;
    L_3 = - y3_dot  +  k_b*y1;

    L_data_1 = y1_data_i - y1;
    L_data_2 = y2_data_i - y2;
    L_data_3 = y3_data_i - y3;

    Loss = [ L_1 ; L_2 ; L_3; L_data_1 ; L_data_2 ; L_data_3];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;

    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);

        %% compute derivatives

        % L1
        L_y1_xi_1 = - c_i*hd   ;
        L_y1_rhs =  h ;

        %L2
        L_y2_xi_2 = - c_i*hd - k_g*(h-h0) ;

        %L3
        L_y3_xi_1 =  k_b*(h-h0)  ;
        L_y3_xi_3 = - c_i*hd ;

        %% Jacobian matrix     

        JJ = [L_y1_xi_1  ,     Z     ,     Z     ,  L_y1_rhs  ; 
                   Z     , L_y2_xi_2 ,     Z     ,      Z     ;
               L_y3_xi_1 ,     Z     , L_y3_xi_3 ,      Z     ;
                -(h-h0)  ,     Z     ,     Z     ,      Z     ;
                   Z     ,  -(h-h0)  ,     Z     ,      Z     ;
                   Z     ,     Z     ,  -(h-h0)  ,      Z     ];

            
        % xi variation
        
        dxi = lsqminnorm(JJ,Loss);
        

        % update xi
        xi = xi - dxi;
        
        xi_1 = xi((0*m)+1:1*m);
        xi_2 = xi((1*m)+1:2*m);
        xi_3 = xi((2*m)+1:3*m);      
        xi_rhs = xi((3*m)+1:4*m);      
        

        %% Re-Build Constrained Expressions
        
        y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;
        y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;
        y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
        rhs = h*xi_rhs;
        
        %% Re-Build the Losses

        L_1 = - y1_dot  +  rhs ;
        L_2 = - y2_dot  -  k_g*y2      ;
        L_3 = - y3_dot  +  k_b*y1;
        L_data_1 = y1_data_i - y1;
        L_data_2 = y2_data_i - y2;
        L_data_3 = y3_data_i - y3;

        Loss = [ L_1 ; L_2 ; L_3; L_data_1 ; L_data_2 ; L_data_3];
        
        l2(2) = norm(Loss);
        iter = iter+1;
        
    end
    
    
%     disp(iter)
    
    
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2)))   ;    
    % Update of constraints
    
    y1_0 = y1(end);
    y2_0 = y2(end);
    y3_0 = y3(end);
       
	sol1((N-1)*(i-1)+1:(N-1)*i+1) = y1;
    sol2((N-1)*(i-1)+1:(N-1)*i+1) = y2;
    sol3((N-1)*(i-1)+1:(N-1)*i+1) = y3;

    rhs_vec((N-1)*(i-1)+1:(N-1)*i+1) = rhs;

    xi_1_vec  = [xi_1_vec ; xi_1];
    xi_2_vec  = [xi_2_vec ; xi_2];
    xi_3_vec  = [xi_3_vec ; xi_3];
    xi_rhs_vec  = [xi_rhs_vec ; xi_rhs];
        
    training_err_vec(i) = training_err;
                     
end

xtfc_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );
fprintf('\n')
fprintf('The average training error for X-TFC is: %g \n', mean(training_err_vec) )

xi_1_mat = reshape(xi_1_vec, m, (n_t-1) );
xi_2_mat = reshape(xi_2_vec, m, (n_t-1) );
xi_3_mat = reshape(xi_3_vec, m, (n_t-1) );
xi_rhs_mat = reshape(xi_rhs_vec, m, (n_t-1) );

sol1_test = [0];
sol2_test = [0];
sol3_test = [0];
rhs_test = [0];

for ii = 1:(n_t-1)

    sol1_test(end) = [];
    sol2_test(end) = [];
    sol3_test(end) = [];
    rhs_test(end) = [];

    sol1_test = [sol1_test ; (h_test-h0_test)*xi_1_mat(:,ii) + y1_initial ];
    y1_initial = sol1_test(end);

    sol2_test = [sol2_test ; (h_test-h0_test)*xi_2_mat(:,ii) + y2_initial ];
    y2_initial = sol2_test(end);

    sol3_test = [sol3_test ; (h_test-h0_test)*xi_3_mat(:,ii) + y3_initial ];
    y3_initial = sol3_test(end);

    rhs_test = [rhs_test ; (h_test)*xi_rhs_mat(:,ii) ];

end

t_test = linspace(t_0,t_f,length(sol1_test));

%% plots

figure(1)
subplot(2,1,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_obs,y1_data_pert,'*','LineWidth',2)
plot(t_obs,y2_data_pert,'*','LineWidth',2)
plot(t_obs,y3_data_pert,'*','LineWidth',2)
plot(t_test,sol1_test,'LineWidth',2)
plot(t_test,sol2_test,'LineWidth',2)
plot(t_test,sol3_test,'LineWidth',2)
ylabel('Tetracycline (mg)')
xlabel('time (hours)')
legend('GI tract (data)', 'Bloodstream (data)', 'Urinary tract (data)', 'GI tract (inferred)', 'Bloodstream (inferred)', 'Urinary tract (inferred)')
box on

subplot(2,1,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(linspace(t_0,t_f,length(rhs_anal)),rhs_anal,'*','LineWidth',2)
plot(t_test,rhs_test,'LineWidth',2)
ylabel('RHS')
xlabel('time (hours)')
legend('k_g*G  -  k_b*B (exact)', 'RHS (inferred)')
box on

sgtitle('Test')



figure(2)
subplot(2,1,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_obs,y1_data_pert,'*','LineWidth',2)
plot(t_obs,y2_data_pert,'*','LineWidth',2)
plot(t_obs,y3_data_pert,'*','LineWidth',2)
plot(t_domain,sol1,'LineWidth',2)
plot(t_domain,sol2,'LineWidth',2)
plot(t_domain,sol3,'LineWidth',2)
ylabel('Tetracycline (mg)')
xlabel('time (hours)')
legend('GI tract (data)', 'Bloodstream (data)', 'Urinary tract (data)', 'GI tract (inferred)', 'Bloodstream (inferred)', 'Urinary tract (inferred)')
box on


subplot(2,1,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(linspace(t_0,t_f,length(rhs_anal)),rhs_anal,'*','LineWidth',2)
plot(t_domain,rhs_vec,'LineWidth',2)
ylabel('RHS')
xlabel('time (hours)')
legend('k_g*G  -  k_b*B (exact)', 'RHS (inferred)')
box on
sgtitle('Training')


err_rhs = abs(rhs_data - rhs_vec);

matrix = [t_domain',  sol1, sol2, sol3, rhs_vec];

filename = 't_B_G_U_f.csv';
writematrix(matrix, filename);


MAE_error = mean(err_rhs);  % Mean Absolute Error
RMSE_error = sqrt(mean(err_rhs.^2));    % Root Mean Square Error
RE_error = 100*norm(err_rhs) / norm(rhs_data);  % Relative Error
fprintf('\n')

fprintf('Mean Absolute Error (MAE) for f(t): %.2e\n', MAE_error);
fprintf('Root Mean Square Error (RMSE) for f(t): %.2e\n', RMSE_error);
fprintf('Relative Error for f(t): %.3g %%\n', RE_error);

