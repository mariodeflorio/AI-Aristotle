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

file_path = 'glucose_insuline_real_1800.csv';
data = readmatrix(file_path);

start = tic;

t_0 = 0; % initial time
t_f = 1800; % final time [min]

N = 6;    % numer of collocation points of the NN
m = 200;    % number of neurons

x = linspace(0,1,N)'; % Discretization of collocation points

t_step = 5;

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

final_subdom = 1; % consider only first sub-domain
% final_subdom = (n_t - 1); % consider all domain

n_points = n_t + (n_t-1)*(N-2);
t_domain = linspace(t_0,t_f,n_points);

type_act = 2; % activation functions

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

options_opt = optimoptions(@lsqnonlin,'MaxIterations',2000,'Algorithm','levenberg-marquardt',...
    'StepTolerance',1e-16,'FunctionTolerance',1e-16,'SpecifyObjectiveGradient',true,'Display','off');

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

V_p = 3; 
V_i = 11;  
V_g = 10; 

t_d = 12;  
k = 1/120;

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

data_to_plot = y3_anal;

t_obs = linspace(t_0,t_f,length(y1_anal));

% Data Perturbation
pert = 0.;
sigma_dist = 3;

y1_noise = sigma_dist*unifrnd(-pert,pert,length(y1_anal),1);
y2_noise = sigma_dist*unifrnd(-pert,pert,length(y2_anal),1);
y3_noise = sigma_dist*unifrnd(-pert,pert,length(y3_anal),1);

y1_anal = y1_anal + y1_noise;
y2_anal = y2_anal + y2_noise;
y3_anal = y3_anal + y3_noise;


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

% Inizialization parameters to infer

E_discover_vec = zeros(n_t-1,1); 
t_p_discover_vec = zeros(n_t-1,1); 
t_i_discover_vec = zeros(n_t-1,1); 
R_m_discover_vec = zeros(n_t-1,1); 
a_1_discover_vec = zeros(n_t-1,1); 

training_err_vec = zeros(n_t-1,1);

tStart = tic;



% initial guess parameters
E = 0.1;
t_p = 5;
t_i = 90;
R_m = 200;
a_1 = 6;

% parameters search range
E_range = [0.1 , 0.3];
t_p_range = [4 , 8];
t_i_range = [60 , 140];
R_m_range = [0.2*209 , 1.8*209];
a_1_range = [0.2*6.6 , 1.8*6.6];

low_range = -inf*ones(6*m + 5,1);
low_range(6*m + 1) = E_range(1) ;
low_range(6*m + 2) = t_p_range(1) ;
low_range(6*m + 3) = t_i_range(1) ;
low_range(6*m + 4) = R_m_range(1) ;
low_range(6*m + 5) = a_1_range(1) ;

up_range = inf*ones(6*m + 5,1);
up_range(6*m + 1) = E_range(2) ;
up_range(6*m + 2) = t_p_range(2) ;
up_range(6*m + 3) = t_i_range(2) ;
up_range(6*m + 4) = R_m_range(2) ;
up_range(6*m + 5) = a_1_range(2) ;

xi_1 = zeros(m,1);
xi_2 = zeros(m,1);
xi_3 = zeros(m,1);
xi_4 = zeros(m,1);
xi_5 = zeros(m,1);
xi_6 = zeros(m,1);


for i = 1:final_subdom

    y1_data_i = y1_data((N-1)*(i-1)+1:(N-1)*i+1) ;    
    y3_data_i = y3_data((N-1)*(i-1)+1:(N-1)*i+1) ;

    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));

    t = linspace(t_tot(i),t_tot(i+1),N)' ;
    
    xi = real([xi_1;xi_2;xi_3;xi_4;xi_5;xi_6;E;t_p;t_i;R_m;a_1]);
    
    xi = lsqnonlin(@fun_lsqnonlin,xi,low_range,up_range,options_opt,t,c_i,...
                    hd,h,h0,m,y1_data_i,y3_data_i,y1_0,y2_0,y3_0,y4_0,y5_0,y6_0,z,Z);

    xi = real(xi);

    xi_1 = xi((0*m)+1:1*m);
    xi_2 = xi((1*m)+1:2*m);
    xi_3 = xi((2*m)+1:3*m);
    xi_4 = xi((3*m)+1:4*m);
    xi_5 = xi((4*m)+1:5*m);
    xi_6 = xi((5*m)+1:6*m);
    E   = xi(6*m+1);
    t_p = xi(6*m+2);
    t_i = xi(6*m+3);
    R_m = xi(6*m+4);
    a_1 = xi(6*m+5);

    %% Re-Build Constrained Expressions

    y1 = real((h-h0)*xi_1 + y1_0);        y1_dot = real(c_i*hd*xi_1);
    y2 = real((h-h0)*xi_2 + y2_0);        y2_dot = real(c_i*hd*xi_2);
    y3 = real((h-h0)*xi_3 + y3_0);        y3_dot = real(c_i*hd*xi_3);
    y4 = real((h-h0)*xi_4 + y4_0);        y4_dot = real(c_i*hd*xi_4);
    y5 = real((h-h0)*xi_5 + y5_0);        y5_dot = real(c_i*hd*xi_5);
    y6 = real((h-h0)*xi_6 + y6_0);        y6_dot = real(c_i*hd*xi_6);

    %% Re-Build the Losses
    sum_IG = 0 ;

    for j = 1 : length(m_j)
        if t(1) >= t_j(j)
            sum_IG = sum_IG + m_j(j)*k*exp(k*(t_j(j) - t));
        end
    end


    f1 = real(R_m./(1 + exp( -y3/V_g/C_1 + a_1 ))); %  insulin secretion
    f2 = U_b*(1 - exp(-y3/V_g/C_2)) ; %  insulin-independent glucose utilization
    f3 = real(( U_0 + U_m ./ (1 + (((1/C_4)*(1./V_i + 1./(E*t_i)))*y2).^(-beta)) ) /V_g/C_3 ); % insulin-dependent glucose utilization
    f4 = R_g./(1 + exp( alpha*(y6/V_p/C_5 - 1 ))) ; % insulin-dependent glucose utilization

    L_1 = - y1_dot  +  f1  -  E*( y1/V_p - y2/V_i) - y1/t_p  ;
    L_2 = - y2_dot  +  E*( y1./V_p - y2./V_i)  -  y2/t_i  ;
    L_3 = - y3_dot  +  f4 + sum_IG  -  f2 -  f3.*y3 ;
    L_4 = - y4_dot  +  (y1 - y4)/t_d ;
    L_5 = - y5_dot  +  (y4 - y5)/t_d ;
    L_6 = - y6_dot  +  (y5 - y6)/t_d ;
    L_data_1 = y1_data_i - y1;
    L_data_3 = y3_data_i - y3;


    Loss = [L_1 ;
        L_2 ;
        L_3 ;
        L_4 ;
        L_5 ;
        L_6 ;
        L_data_1 ;
        L_data_3
        ];


    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2))) + ...
        sqrt(mean(abs(L_4.^2))) +  sqrt(mean(abs(L_5.^2))) +  sqrt(mean(abs(L_6.^2)))  ;
    
    % Update of constraints

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

    training_err_vec(i) = training_err;
    
    E_discover_vec(i) = E ;
    t_p_discover_vec(i) = t_p ;
    t_i_discover_vec(i) = t_i ;
    R_m_discover_vec(i) = R_m ;
    a_1_discover_vec(i) = a_1 ;

    fprintf('\n')
    fprintf('\n')
    fprintf(' Subdomain %.0f \n', i )    
    fprintf('\n')
    fprintf(' The discovered value of the parameter E is: %12.12f \n', mean(E) )
    fprintf(' The discovered value of the parameter t_p is: %12.12f \n', mean(t_p) )
    fprintf(' The discovered value of the parameter t_i is: %12.12f \n', mean(t_i) )
    fprintf(' The discovered value of the parameter R_m is: %12.12f \n', mean(R_m) )
    fprintf(' The discovered value of the parameter a_1 is: %12.12f \n', mean(a_1) )
    fprintf('\n')
    fprintf(' Relative error for E is  : %.3g %%\n', 100*abs(E - 0.2)/0.2 )
    fprintf(' Relative error for t_p is: %.3g %%\n', 100*abs(t_p - 6)/6 )
    fprintf(' Relative error for t_i is: %.3g %%\n', 100*abs(t_i - 100)/100 )
    fprintf(' Relative error for R_m is: %.3g %%\n', 100*abs(R_m - 209)/209 )
    fprintf(' Relative error for a_1 is: %.3g %%\n', 100*abs(a_1 - 6.6)/6.6 )
    fprintf('\n')
    fprintf('======================================================================')
    fprintf('\n')

end

%%

xtfc_elapsedtime = toc(tStart) ;
fprintf('\n')

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );
fprintf('\n')



%%
function [Loss,JJ] = fun_lsqnonlin(xi,t,c_i,hd,h,h0,m,y1_data_i,y3_data_i,y1_0,y2_0,y3_0,y4_0,y5_0,y6_0,z,Z)

V_p = 3; 
V_i = 11;  
V_g = 10; 

t_d = 12;  

k = 1/120;

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


xi_1 = xi((0*m)+1:1*m);
xi_2 = xi((1*m)+1:2*m);
xi_3 = xi((2*m)+1:3*m);
xi_4 = xi((3*m)+1:4*m);
xi_5 = xi((4*m)+1:5*m);
xi_6 = xi((5*m)+1:6*m);
E = xi(6*m+1);
t_p = xi(6*m+2);
t_i = xi(6*m+3);
R_m = xi(6*m+4);
a_1 = xi(6*m+5);


y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;
y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;
y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
y4 = (h-h0)*xi_4 + y4_0;        y4_dot = c_i*hd*xi_4;
y5 = (h-h0)*xi_5 + y5_0;        y5_dot = c_i*hd*xi_5;
y6 = (h-h0)*xi_6 + y6_0;        y6_dot = c_i*hd*xi_6;

% compute derivatives

% L1
L_y1_xi_1 = - c_i*hd - (h - h0)./t_p - (E.*(h - h0))./V_p ;
L_y1_xi_2 = (E*(h - h0))./V_i;
L_y1_xi_3 = (R_m*exp(a_1 - y3/(C_1*V_g)).*(h - h0))./(C_1*V_g*(exp(a_1 - y3./(C_1*V_g)) + 1).^2) ;
L_y1_E = y2./V_i - y1./V_p ;
L_y1_t_p = y1./(t_p.^2); 
L_y1_R_m = 1./(exp(a_1 - y3./(C_1*V_g)) + 1);
L_y1_a_1 = -(R_m.*exp(a_1 - y3./(C_1*V_g)))./(exp(a_1 - y3./(C_1*V_g)) + 1).^2;


%L2
L_y2_xi_1 =  (E.*(h - h0))./V_p ;
L_y2_xi_2 = - c_i*hd - (h - h0)./t_i - (E.*(h - h0))./V_i    ;
L_y2_E = y1./V_p - y2./V_i ;
L_y2_t_i = y2./t_i.^2 ;

%L3
L_y3_xi_2 = -(U_m*beta.*y3.*(h - h0).*(1/V_i + 1./(E*t_i)))./(C_3*C_4*V_g.*(1./((y2.*(1/V_i + 1./(E*t_i)))./C_4).^beta + 1).^2.*((y2.*(1/V_i + 1/(E*t_i)))./C_4).^(beta + 1));
L_y3_xi_3 = - c_i*hd - ((U_0 + U_m./(1./((y2.*(1/V_i + 1/(E*t_i)))./C_4).^beta + 1)).*(h - h0))./(C_3*V_g) - (U_b*exp(-y3./(C_2*V_g)).*(h - h0))./(C_2*V_g);
L_y3_xi_6 = -(R_g*alpha*exp(alpha*(y6./(C_5*V_p) - 1)).*(h - h0))./(C_5*V_p*(exp(alpha*(y6./(C_5*V_p) - 1)) + 1).^2);
L_y3_E = (U_m*beta*y2.*y3)/(C_3*C_4*E.^2*V_g*t_i*(1/((y2.*(1./V_i + 1./(E*t_i)))./C_4).^beta + 1).^2*((y2.*(1./V_i + 1/(E*t_i)))./C_4).^(beta + 1));
L_y3_t_i = (U_m*beta*y2.*y3)./(C_3*C_4*E*V_g*t_i.^2.*(1/((y2.*(1./V_i + 1./(E*t_i)))./C_4).^beta + 1).^2*((y2.*(1./V_i + 1./(E*t_i)))./C_4).^(beta + 1));


%L4
L_y4_xi_1 = (h-h0)./t_d ;
L_y4_xi_4 =  - c_i*hd  - (h-h0)./t_d ;

%L5
L_y5_xi_4 = (h-h0)./t_d ;
L_y5_xi_5 =  - c_i*hd  - (h-h0)./t_d ;

%L6
L_y6_xi_5 = (h-h0)./t_d ;
L_y6_xi_6 =  - c_i*hd  - (h-h0)./t_d ;


%% Jacobian matrix

JJ = [ L_y1_xi_1 , L_y1_xi_2 , L_y1_xi_3 ,     Z     ,     Z     ,     Z     ,  L_y1_E  ,  L_y1_t_p  ,     z      ,  L_y1_R_m  ,  L_y1_a_1  ; 
       L_y2_xi_1 , L_y2_xi_2 ,     Z     ,     Z     ,     Z     ,     Z     ,  L_y2_E  ,      z     ,  L_y2_t_i  ,     z      ,     z      ;
           Z     , L_y3_xi_2 , L_y3_xi_3 ,     Z     ,     Z     , L_y3_xi_6 ,  L_y3_E  ,      z     ,  L_y3_t_i  ,     z      ,     z      ;
       L_y4_xi_1 ,     Z     ,     Z     , L_y4_xi_4 ,     Z     ,     Z     ,     z    ,      z     ,     z      ,     z      ,     z      ;
           Z     ,     Z     ,     Z     , L_y5_xi_4 , L_y5_xi_5 ,     Z     ,     z    ,      z     ,     z      ,     z      ,     z      ; 
           Z     ,     Z     ,     Z     ,     Z     , L_y6_xi_5 , L_y6_xi_6 ,     z    ,      z     ,     z      ,     z      ,     z      ;
       -(h-h0)   ,     Z     ,     Z     ,     Z     ,     Z     ,     Z     ,     z    ,      z     ,     z      ,     z      ,     z      ;
           Z     ,     Z     ,  -(h-h0)  ,     Z     ,     Z     ,     Z     ,     z    ,      z     ,     z      ,     z      ,     z      ];

%% Build the Losses

sum_IG = 0 ;

for j = 1 : length(m_j)
    if t(1) >= t_j(j)
        sum_IG = sum_IG + m_j(j)*k*exp(k*(t_j(j) - t));
    end
end

f1 = R_m./(1 + exp( -y3/V_g/C_1 + a_1 )); %  insulin secretion
f2 = U_b*(1 - exp(-y3/V_g/C_2)) ; %  insulin-independent glucose utilization
f3 = ( U_0 + U_m ./ (1 + (((1/C_4)*(1/V_i + 1/(E*t_i)))*y2).^(-beta)) ) /V_g/C_3 ; % insulin-dependent glucose utilization
f4 = R_g./(1 + exp( alpha*(y6/V_p/C_5 - 1 ))) ; % insulin-dependent glucose utilization

L_1 = - y1_dot  +  f1  -  E*( y1/V_p - y2/V_i) - y1/t_p  ;
L_2 = - y2_dot  +  E*( y1/V_p - y2/V_i)  -  y2/t_i  ;
L_3 = - y3_dot  +  f4  -  f2 -  f3.*y3 ;
L_4 = - y4_dot  +  (y1 - y4)/t_d ;
L_5 = - y5_dot  +  (y4 - y5)/t_d ;
L_6 = - y6_dot  +  (y5 - y6)/t_d ;
L_data_1 = y1_data_i - y1;
L_data_3 = y3_data_i - y3;


Loss = [L_1 ;
    L_2 ;
    L_3 ;
    L_4 ;
    L_5 ;
    L_6 ;
    L_data_1 ;
    L_data_3
    ];
end




