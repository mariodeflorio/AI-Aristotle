%%
clear; close all; clc;
% format long
%--------------------------------------------------------------------------
%{ 
  X-TFC applied to Gray-Box Systems Biology
  Test Case - Ultradian endocrine model

  Author:
  Mario De Florio
%}
%%
%--------------------------------------------------------------------------
%% Input

rng('default') % set random seed

file_path = 'glucose_insuline_real_1800.csv';
data = readmatrix(file_path);

start = tic;

t_0 = 0; % initial time
t_f = 1800; % final time [min]

N = 6;    % numer of collocation points of the NN
m = 30;    % number of neurons

x = linspace(0,1,N)'; % Discretization of collocation points

t_step = 5;

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

n_points = n_t + (n_t-1)*(N-2);
t_domain = linspace(t_0,t_f,n_points);

% iterative least-square parameters

IterMax = 100;
IterTol = 1e-9;

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

V_p = 3; % volume of insulin distribution in the plasma
V_i = 11; % volume of the remote insulin compartment
V_g = 10; % volume of the glucose space

E = 0.2; % rate constant for exchange of insulin between the plasma and remote compartments

t_p = 6; % time constant for plasma insulin degradation
t_i = 100; % time constant for remote insulin degradation
t_d = 12; % delay time between plasma insulin and glucose production

k = 1/120;

R_m = 209; 
a_1 = 6.6;
C_1 = 300;
C_2 = 144;
C_3 = 100;
C_4 = 80;
C_5 = 26;

U_b = 72;
U_0 = 4;
U_m = 90;
R_g = 180;

alpha = 7.5;
beta = 1.772;

t_j = [300, 650, 1100] ;
m_j = 1e3*[60, 40, 50] ;

%% =======================================
% synthetic data (EXACT SOLUTION)

y1_anal = data(:,2);
y2_anal = data(:,3);
y3_anal = data(:,4);
rhs_1 = data(:,5);
rhs_2 = data(:,6);

t_obs = linspace(t_0,t_f,length(y1_anal));

%% Data Perturbation

noise_lev = 0.0;

y1_data_pert = y1_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y1_anal) , 1));
y2_data_pert = y2_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y2_anal) , 1));
y3_data_pert = y3_anal .* ( 1 + noise_lev*unifrnd(-1,1,length(y3_anal) , 1));

%% interpolation observed data

y_RK_inter = 1:length(y1_data_pert);
ind = linspace(1,length(y1_data_pert),n_points);

y1_data = spline(y_RK_inter,y1_data_pert,ind)';
y2_data = spline(y_RK_inter,y2_data_pert,ind)';
y3_data = spline(y_RK_inter,y3_data_pert,ind)';
rhs_1_data = spline(y_RK_inter,rhs_1,ind)';
rhs_2_data = spline(y_RK_inter,rhs_2,ind)';

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


%% Ax=b construction
Z = zeros(N,m);
z = zeros(N,1);

% Initial Values

y1_initial = 12*3; 
y2_initial = 4*11; 
y3_initial = 110*(10^2);
y4_initial = 0; 
y5_initial = 0; 
y6_initial = 0;

y1_0 = y1_initial;
y2_0 = y2_initial;
y3_0 = y3_initial;
y4_0 = y4_initial;
y5_0 = y5_initial;
y6_0 = y6_initial;

sol1 = zeros(n_points,1); % I_p
sol2 = zeros(n_points,1); % I_i
sol3 = zeros(n_points,1); % G
sol4 = zeros(n_points,1); % h_1
sol5 = zeros(n_points,1); % h_2
sol6 = zeros(n_points,1); % h_3

rhs_1_vec = zeros(n_points,1); % 
rhs_2_vec = zeros(n_points,1); % 

IG = zeros(n_points,1); % I_G

tStart = tic;

for i = 1:(n_t-1)

    xi_1 = zeros(m,1);
    xi_2 = zeros(m,1);
    xi_3 = zeros(m,1);
    xi_4 = zeros(m,1);
    xi_5 = zeros(m,1);
    xi_6 = zeros(m,1);
    xi_rhs_1 = zeros(m,1);
    xi_rhs_2 = zeros(m,1);

    y1_data_i = y1_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y3_data_i = y3_data((N-1)*(i-1)+1:(N-1)*i+1) ;

    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));

    t = linspace(t_tot(i),t_tot(i+1),N)' ;
    
    xi = [xi_1;xi_2;xi_3;xi_4;xi_5;xi_6; xi_rhs_1;xi_rhs_2];

    %% Build Constrained Expressions
    
    y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;     
    y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;   
    y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
    y4 = (h-h0)*xi_4 + y4_0;        y4_dot = c_i*hd*xi_4;     
    y5 = (h-h0)*xi_5 + y5_0;        y5_dot = c_i*hd*xi_5;   
    y6 = (h-h0)*xi_6 + y6_0;        y6_dot = c_i*hd*xi_6;
    rhs_1 = (h)*xi_rhs_1;
    rhs_2 = (h)*xi_rhs_2;
   
    %% Build the Losses  

    sum_IG = 0 ;

    for j = 1 : length(m_j)
        if t(1) >= t_j(j)
            sum_IG = sum_IG + m_j(j)*k*exp(k*(t_j(j) - t));
        end
    end

    f1 = R_m./(1 + exp( -y3/V_g/C_1 + a_1 )); %  insulin secretion
    f2 = U_b*(1 - exp(-y3/V_g/C_2)) ; %  insulin-independent glucose utilization
    f3 = ( U_0 + U_m ./ (1 + (((1/C_4)*(1./V_i + 1./(E*t_i)))*y2).^(-beta)) ) /V_g/C_3 ; % insulin-dependent glucose utilization
    f4 = R_g./(1 + exp( alpha*(y6/V_p/C_5 - 1 ))) ; % insulin-dependent glucose utilization

    L_1 = - y1_dot  +  f1  + rhs_1  ;
    L_2 = - y2_dot  +  rhs_2  ;
    L_3 = - y3_dot  +  f4  +  sum_IG  -  f2 -  f3.*y3 ;
    L_4 = - y4_dot  +  (y1 - y4)/t_d ;
    L_5 = - y5_dot  +  (y4 - y5)/t_d ;
    L_6 = - y6_dot  +  (y5 - y6)/t_d ;
    L_data_1 = y1_data_i - y1;
    L_data_3 = y3_data_i - y3;

    Loss = [L_1 ; 
            L_2 ; 
            L_3; 
            L_4 ; 
            L_5 ; 
            L_6 ; 
            L_data_1 ; 
            L_data_3];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);

        %% compute derivatives

        % L1
        L_y1_xi_1 = - c_i*hd   ;
        L_y1_xi_3 = (R_m*exp(a_1 - y3/(C_1*V_g)).*(h - h0))./(C_1*V_g*(exp(a_1 - y3./(C_1*V_g)) + 1).^2) ;
        L_y1_rhs_1 = h;

        %L2
        L_y2_xi_2 = - c_i*hd  ;
        L_y2_rhs_2 = h;
               
        %L3
        L_y3_xi_2 = -(U_m*beta.*y3.*(h - h0).*(1/V_i + 1./(E*t_i)))./(C_3*C_4*V_g.*(1./((y2.*(1/V_i + 1./(E*t_i)))./C_4).^beta + 1).^2.*((y2.*(1/V_i + 1/(E*t_i)))./C_4).^(beta + 1));
        L_y3_xi_3 = - c_i*hd - ((U_0 + U_m./(1./((y2.*(1/V_i + 1/(E*t_i)))./C_4).^beta + 1)).*(h - h0))./(C_3*V_g) - (U_b*exp(-y3./(C_2*V_g)).*(h - h0))./(C_2*V_g);
        L_y3_xi_6 = -(R_g*alpha*exp(alpha*(y6./(C_5*V_p) - 1)).*(h - h0))./(C_5*V_p*(exp(alpha*(y6./(C_5*V_p) - 1)) + 1).^2);

        %L4
        L_y4_xi_1 = (h-h0)./t_d ;
        L_y4_xi_4 =  - c_i*hd  - (h-h0)./t_d ;

        %L5
        L_y5_xi_4 = (h-h0)./t_d ;
        L_y5_xi_5 =  - c_i*hd  - (h-h0)./t_d ;

        %L6
        L_y6_xi_5 = (h-h0)./t_d ;
        L_y6_xi_6 =  - c_i*hd  - (h-h0)./t_d ;

        % Jacobian matrix
        JJ = [ L_y1_xi_1 ,     Z     , L_y1_xi_3 ,     Z     ,     Z     ,     Z     ,  L_y1_rhs_1  ,      Z       ;
                   Z     , L_y2_xi_2 ,     Z     ,     Z     ,     Z     ,     Z     ,      Z       ,  L_y2_rhs_2  ;
                   Z     , L_y3_xi_2 , L_y3_xi_3 ,     Z     ,     Z     , L_y3_xi_6 ,      Z       ,      Z       ;
               L_y4_xi_1 ,     Z     ,     Z     , L_y4_xi_4 ,     Z     ,     Z     ,      Z       ,      Z       ;
                   Z     ,     Z     ,     Z     , L_y5_xi_4 , L_y5_xi_5 ,     Z     ,      Z       ,      Z       ;
                   Z     ,     Z     ,     Z     ,     Z     , L_y6_xi_5 , L_y6_xi_6 ,      Z       ,      Z       ;
                -(h-h0)  ,     Z     ,     Z     ,     Z     ,     Z     ,     Z     ,      Z       ,      Z       ;
                   Z     ,     Z     ,  -(h-h0)  ,     Z     ,     Z     ,     Z     ,      Z       ,      Z       ];                        

            
        % xi variation
        dxi = lsqminnorm(JJ,Loss);
        
        % update xi
        xi = xi - dxi;
        
        xi_1 = xi((0*m)+1:1*m);
        xi_2 = xi((1*m)+1:2*m);
        xi_3 = xi((2*m)+1:3*m);
        xi_4 = xi((3*m)+1:4*m);
        xi_5 = xi((4*m)+1:5*m);
        xi_6 = xi((5*m)+1:6*m);
        xi_rhs_1 = xi((6*m)+1:7*m);
        xi_rhs_2 = xi((7*m)+1:8*m);
        

        %% Re-Build Constrained Expressions
        
        y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;
        y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;
        y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
        y4 = (h-h0)*xi_4 + y4_0;        y4_dot = c_i*hd*xi_4;
        y5 = (h-h0)*xi_5 + y5_0;        y5_dot = c_i*hd*xi_5;
        y6 = (h-h0)*xi_6 + y6_0;        y6_dot = c_i*hd*xi_6;
        rhs_1 = (h)*xi_rhs_1;
        rhs_2 = (h)*xi_rhs_2;
        
        %% Re-Build the Losses

        f1 = R_m./(1 + exp( -y3/V_g/C_1 + a_1 )); %  insulin secretion
        f2 = U_b*(1 - exp(-y3/V_g/C_2)) ; %  insulin-independent glucose utilization
        f3 = ( U_0 + U_m ./ (1 + (((1/C_4)*(1./V_i + 1./(E*t_i)))*y2).^(-beta)) ) /V_g/C_3 ; % insulin-dependent glucose utilization
        f4 = R_g./(1 + exp( alpha*(y6/V_p/C_5 - 1 ))) ; % insulin-dependent glucose utilization

        L_1 = - y1_dot  +  f1  +  rhs_1  ;
        L_2 = - y2_dot  +  rhs_2  ;
        L_3 = - y3_dot  +  f4  +  sum_IG  -  f2 -  f3.*y3 ;
        L_4 = - y4_dot  +  (y1 - y4)/t_d ;
        L_5 = - y5_dot  +  (y4 - y5)/t_d ;
        L_6 = - y6_dot  +  (y5 - y6)/t_d ;
        L_data_1 = y1_data_i - y1;
        L_data_3 = y3_data_i - y3;

        Loss = [L_1 ; 
                L_2 ; 
                L_3; 
                L_4 ; 
                L_5 ; 
                L_6 ; 
                L_data_1 ; 
                L_data_3];        
            
        l2(2) = norm(Loss);
        iter = iter+1;
        
    end
    
    y1_0 = y1(end);
    y2_0 = y2(end);
    y3_0 = y3(end);
    y4_0 = y4(end);
    y5_0 = y5(end);
    y6_0 = y6(end);
        
	sol1((N-1)*(i-1)+1:(N-1)*i+1) = y1;
    sol2((N-1)*(i-1)+1:(N-1)*i+1) = y2;
    sol3((N-1)*(i-1)+1:(N-1)*i+1) = y3;
    sol4((N-1)*(i-1)+1:(N-1)*i+1) = y4;
    sol5((N-1)*(i-1)+1:(N-1)*i+1) = y5;
    sol6((N-1)*(i-1)+1:(N-1)*i+1) = y6;
    rhs_1_vec((N-1)*(i-1)+1:(N-1)*i+1) = rhs_1;
    rhs_2_vec((N-1)*(i-1)+1:(N-1)*i+1) = rhs_2;

    IG((N-1)*(i-1)+1:(N-1)*i+1) = sum_IG;

              
end

xtfc_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );

%% plots

figure(1)
subplot(2,2,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_domain,sol1,'LineWidth',2)
plot(t_obs,y1_data_pert,'o')
ylabel('I_p (\muU/ml)')
legend('Inferred data','Observed data')
box on

subplot(2,2,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_domain,sol2,'LineWidth',2)
% plot(t_obs,y2_data_pert,'o')
ylabel('I_i (\muU/ml)')
box on

subplot(2,2,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_domain,sol3,'LineWidth',2)
plot(t_obs,y3_data_pert,'o')
ylabel('G (mg/dl)')
box on




figure(2)
subplot(1,2,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_domain,rhs_1_data,'LineWidth',2)
plot(t_domain,rhs_1_vec,'--','LineWidth',2)
ylabel('f(t)')
xlabel('time (minutes)')
legend('Exact missing term','Inferred missing term')
box on

subplot(1,2,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_domain,rhs_2_data,'LineWidth',2)
plot(t_domain,rhs_2_vec,'--','LineWidth',2)
ylabel('g(t)')
xlabel('time (minutes)')
box on

matrix = [t_domain',sol1, sol2, sol3, rhs_1_vec, rhs_2_vec];

filename = 't_Ip_Ii_G_f_g.csv';
writematrix(matrix, filename);

err_rhs_1 = abs(rhs_1_data - rhs_1_vec);
err_rhs_2 = abs(rhs_2_data - rhs_2_vec);


MAE_error_1 = mean(err_rhs_1);  % Mean Absolute Error
RMSE_error_1 = sqrt(mean(err_rhs_1.^2));    % Root Mean Square Error
RE_error_1 = norm(err_rhs_1) / norm(rhs_1_data);  % Relative Error
fprintf('\n')

fprintf('Mean Absolute Error (MAE) for f(t): %.2e\n', MAE_error_1);
fprintf('Root Mean Square Error (RMSE) for f(t): %.2e\n', RMSE_error_1);
fprintf('Relative Error for f(t): %.2e %%\n', 100*RE_error_1);


MAE_error_2 = mean(err_rhs_2);  % Mean Absolute Error
RMSE_error_2 = sqrt(mean(err_rhs_2.^2));     % Root Mean Square Error
RE_error_2 = norm(err_rhs_2) / norm(rhs_2_data);  % Relative Error
fprintf('\n')
fprintf('Mean Absolute Error (MAE) for g(t): %.2e\n', MAE_error_2);
fprintf('Root Mean Square Error (RMSE) for g(t): %.2e\n', RMSE_error_2);
fprintf('Relative Error for g(t): %.2e %%\n', 100*RE_error_2);
fprintf('\n')
% fprintf('%.2e & %.2e & %.2e & %.2e & %.2e & %.2e \n', MAE_error_1,   RMSE_error_1,  100*RE_error_1, MAE_error_2,   RMSE_error_2,  100*RE_error_2);



