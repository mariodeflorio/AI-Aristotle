import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import optax
import time
import pandas as pd
import jaxopt
from scipy.integrate import odeint

##################### Solver for the Exact solution ##################
def drug_model(
    t,
    kg = 0.72,
    kb = 0.15,
    G0 = 0.1,
):
    def func(y, t):
        G, B, U = y[0], y[1], y[2]

        return [
            - kg * G,
            kg * G - kb * B,
            kb * B,
            ]

    y0 = [G0, 0, 0]
    return odeint(func, y0, t)


t_dense = jnp.linspace(0, 50, 501)[:, None]
y_dense = drug_model(np.ravel(t_dense))


sample_rate = 5
dp=  500/sample_rate
#t_data = t_dense[::sample_rate,0:1]
#G_data = y_dense[::sample_rate,0:1]
#B_data = y_dense[::sample_rate,1:2]
#U_data = y_dense[::sample_rate,2:3]



######################################################################
noise = 0.04

# Load the first file
file_path_1 = F'./noise_{noise}.csv'  # Replace with your file path
data_noisy = pd.read_csv(file_path_1)
t_data = data_noisy['t']
t_data = t_data.to_numpy(dtype=np.float32).reshape(-1, 1)

# Concatenate the noisy data
data = jnp.column_stack((data_noisy['G'],data_noisy['B'],data_noisy['U']))

#initial Condition
t_i  = jnp.array([[0]])
IC = np.array([data_noisy['G'][0], 0, 0])
IC = np.array(IC).astype(np.float32)



####################### Neural Network  ##############################

tmin, tmax = t_dense[0,0], t_dense[-1,0]

#t = (t-np.min(t))/(np.max(t)-np.min(t))
def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(5678), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(key, shape=(n_in, n_out)) / jnp.sqrt(n_in) #random initialization
        B = jax.random.normal(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params


def fwd(params,t):
  X = jnp.concatenate([t],axis=1)
  *hidden,last = params
  for layer in hidden :
    X = jax.nn.tanh(X@layer['W']+layer['B'])
    # X = jnp.sin(X@layer['W']+layer['B'])
  return X@last['W'] + last['B']

def fwd_extra(params,t):
  X = jnp.concatenate([t],axis=1)
  *hidden,last = params
  for layer in hidden :
    X = jax.nn.tanh(X@layer['W']+layer['B'])
    # X = jnp.sin(X@layer['W']+layer['B'])

  X = X@last['W'] + last['B']
  # X = X**2
  return X

######################################################################

@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)



#_______   Physics loss  ___________

def ODE_loss(t, y1, y2, y3, f_t):
    kg = 0.72
    kb = 0.15
    G0 = 0.1
    # f_t=kg * y1(t) + kb * y2(t)

    y1_t = lambda t: jax.grad(lambda t: jnp.sum(y1(t)))(t)
    y2_t = lambda t: jax.grad(lambda t: jnp.sum(y2(t)))(t)
    y3_t = lambda t: jax.grad(lambda t: jnp.sum(y3(t)))(t)
    ode1 = y1_t(t) + kg * y1(t)
    ode2 = y2_t(t) - f_t(t)
    ode3 = y3_t(t) - kb * y2(t)

    return ode1, ode2, ode3
#___________________________________



#collocation points
N_c = 500

t_c = jnp.linspace(tmin, tmax, N_c+1)[:, None]

def loss_fun(params, params_extra,l1 ,l2 ,l3 , t_i, t_d, t_c, data_IC, data):

    y1_func = lambda t: fwd(params, t)[:, [0]]
    y2_func = lambda t: fwd(params, t)[:, [1]]
    y3_func = lambda t: fwd(params, t)[:, [2]]

    f_t     = lambda t: fwd_extra(params_extra, t)[:, [0]]

    loss_y1, loss_y2, loss_y3 = ODE_loss(t_c, y1_func, y2_func, y3_func, f_t)

    loss_y1 = l1*loss_y1
    loss_y2 = l2*loss_y2
    loss_y3 = l3*loss_y3

    loss_ode1 = jnp.mean(loss_y1 ** 2)
    loss_ode2 = jnp.mean(loss_y2 ** 2)
    loss_ode3 = jnp.mean(loss_y3 ** 2)


    # Compute the loss for the initial conditions
    t_i = t_i.flatten()[:,None]
    pred_IC = jnp.concatenate([y1_func(t_i), y2_func(t_i), y3_func(t_i)],axis=1)
    loss_IC = MSE(data_IC, pred_IC)

    # Compute the loss for Y_data
    t_d    = t_d.flatten()[:,None]
    pred_d = jnp.concatenate([y1_func(t_d), y2_func(t_d), y3_func(t_d)],axis=1)
    loss_data = MSE(data, pred_d)


    return loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3


def loss_fun_total(params, params_extra,l1,l2, l3, t_i, t_d, t_c, data_IC, data, loss_weight):

    loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3 = loss_fun(params, params_extra,l1,l2, l3, t_i, t_d, t_c, data_IC, data)

    loss_total = loss_weight[0]*loss_IC+ loss_weight[1]*loss_data\
                + loss_weight[2]*loss_ode1+ loss_weight[3]*loss_ode2\
                + loss_weight[4]*loss_ode3

    return loss_total


def loss_fun_total_lbfgs(params_total, t_i, t_d, t_c, data_IC, data, loss_weight):

    params, params_extra, l1,l2, l3 = params_total[0], params_total[1], params_total[2], params_total[3], params_total[4]

    loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3 = loss_fun(params, params_extra,l1 ,l2, l3, t_i, t_d, t_c, data_IC, data)


    loss_total = loss_weight[0]*loss_IC+ loss_weight[1]*loss_data\
                + loss_weight[2]*loss_ode1+ loss_weight[3]*loss_ode2\
                + loss_weight[4]*loss_ode3

    print(f"Loss: {loss_total}")

    return loss_total




@jax.jit
def update(opt_state, opt_state_extra, params, params_extra, params_l1, params_l2, params_l3, opt_state_l1,opt_state_l2, opt_state_l3, t_i, t_data, t_c, IC, data, loss_weight):
  grads=jax.grad(loss_fun_total, argnums=[0,1,2,3,4])(params, params_extra,params_l1, params_l2, params_l3, t_i, t_data, t_c, IC, data, loss_weight)

  #Update params
  updates, opt_state = optimizer.update(grads[0], opt_state)
  params = optax.apply_updates(params, updates)

  updates_extra, opt_state_extra = optimizer_ex.update(grads[1], opt_state_extra)
  params_extra = optax.apply_updates(params_extra, updates_extra)

  updates_l1, opt_state_l1 = optimizer.update(-grads[2], opt_state_l1)
  params_l1 = optax.apply_updates(params_l1, updates_l1)

  updates_l2, opt_state_l2 = optimizer.update(-grads[3], opt_state_l2)
  params_l2 = optax.apply_updates(params_l2, updates_l2)

  updates_l3, opt_state_l3 = optimizer.update(-grads[4], opt_state_l3)
  params_l3 = optax.apply_updates(params_l3, updates_l3)

  return opt_state,params,opt_state_extra,params_extra, params_l1, params_l2, params_l3, opt_state_l1,opt_state_l2, opt_state_l3




###################### defining NN architecture ######################

params = init_params([1] + [50]*6+[3])
params_extra = init_params([1] + [20]*4+[1])  # Initialize parameters for the extra neural network

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
optimizer_ex = optax.adam(1e-4)
opt_state_extra = optimizer_ex.init(params_extra)
######################################################################


# Initializing the adaptive loss weights

keys = jax.random.split(jax.random.PRNGKey(5678), 10)
lambda_1 = jax.random.uniform(keys[0], shape=(N_c + 1, 1))
lambda_2 = jax.random.uniform(keys[1], shape=(N_c + 1, 1))
lambda_3 = jax.random.uniform(keys[2], shape=(N_c + 1, 1))

opt_state_l1 = optimizer.init(lambda_1)
opt_state_l2 = optimizer.init(lambda_2)
opt_state_l3 = optimizer.init(lambda_3)



#################### Two Step training  ##############################

start_time = time.time()
epochs_phase1 = 5000
epochs_phase2 = 30000

loss_weight_phase1 = [1, 1, 0, 0, 0]
loss_weight_phase2 = [1, 1, 1, 1, 1]

loss_his, loss_indi_his, epoch_his = [], [], []

for ep in range(epochs_phase1 + epochs_phase2 + 1):
    if ep <= epochs_phase1:
        loss_weight = loss_weight_phase1
    else:
        loss_weight = loss_weight_phase2

    opt_state,params,opt_state_extra,\
    params_extra, params_l1, params_l2,\
    params_l3, opt_state_l1,opt_state_l2, opt_state_l3 = update(opt_state, opt_state_extra, params, params_extra,\
                                                             lambda_1, lambda_2, lambda_3, opt_state_l1,\
                                                              opt_state_l2, opt_state_l3, t_i, t_data, t_c, IC, data, loss_weight)

    # print loss and epoch info
    if ep %(1000) ==0:
      loss_val = loss_fun_total(params, params_extra, params_l1, params_l2, params_l3, t_i, t_data, t_c, IC, data, loss_weight)
      loss_val_individual = loss_fun(params, params_extra,params_l1, params_l2, params_l3, t_i, t_data, t_c, IC, data)
      epoch_his.append(ep)
      loss_his.append(loss_val)
      loss_indi_his.append(loss_val_individual)
      print(f'Epoch={ep}, \t loss={loss_val:.3e}, \t loss_IC={loss_val_individual[0]:.3e}, \t loss_d={loss_val_individual[1]:.3e}, \t loss_e1={loss_val_individual[2]:.3e}, \t loss_e2={loss_val_individual[3]:.3e}, \t loss_e3={loss_val_individual[4]:.3e}')
      end_time = time.time()

      running_time = end_time - start_time
      print(f"Total running time: {running_time:.4f} seconds")


####################   LBFGS OPTIMIZER. ##############################
print("L-BFGS optimizer begins")

loss_weight = [1, 1, 1, 1, 1]

params_total = []
params_total.append(params)
params_total.append(params_extra)
params_total.append(params_l1)
params_total.append(params_l2)
params_total.append(params_l3)

start_time = time.time()

l_fun = lambda p_l: loss_fun_total_lbfgs(p_l, t_i, t_data, t_c, IC, data, loss_weight)

optimizer_lbfgs = jaxopt.LBFGS(fun=l_fun, history_size=100,  verbose=True, maxiter=100)
sol = optimizer_lbfgs.run(params_total)
params_total = sol.params

end_time = time.time()

running_time += end_time - start_time
print(f"Total running time: {running_time:.4f} seconds")


params = params_total[0]
params_extra = params_total[1]


######################################################################
# Saving the model
np.savez(f'./params_{noise}.npz', *params)
np.savez(f'./params_extra_{noise}.npz', *params_extra)

# defining Exact form of missing part
kg = 0.72
kb = 0.15
G0 = 0.1
f_t_analytical = kg * y_dense[:, 0] - kb * y_dense[:, 1]

# output of second NN for missing part
f_t_neural = fwd_extra(params_extra, t_dense)[:, 0]

# output of main neural network
pred = fwd(params,t_dense)
G_pred = pred[:,0:1]
B_pred = pred[:,1:2]
U_pred = pred[:,2:3]
df_f = pd.DataFrame({"t": np.ravel(t_dense),"G": np.ravel(G_pred), "B": np.ravel(B_pred),"U": np.ravel(U_pred) ,"ft": np.ravel(f_t_neural) })
df_f.to_csv(f"./pred_500_{noise}.csv", index=False)



##################### plots ##########################################

plt.figure()
plt.plot(t_dense, y_dense[:, 0:1],'-b',label='G (true)')
plt.plot(t_dense, y_dense[:, 1:2],'-r',label='B (true)')
plt.plot(t_dense, y_dense[:, 2:3],'-g',label='U (true)')

plt.scatter(t_data, data[:,0:1], label='G data points', s=20)
plt.scatter(t_data, data[:,1:2], label='B data points', s=20)
plt.scatter(t_data, data[:,2:3], label='U data points', s=20)


plt.plot(t_dense,pred[:,0],'--k',label='PINN',linewidth=2)
plt.plot(t_dense,pred[:,1],'--k',linewidth=2)
plt.plot(t_dense,pred[:,2],'--k',linewidth=2)
plt.title(f'Number of data point: {noise}')
plt.xlabel('Time (hr)')
plt.ylabel('Tetracycline (ng)')
plt.legend()
plt.savefig(f'./Pred__noise_{noise}.png', dpi=300)



plt.figure()

plt.plot(t_dense, f_t_analytical, label='$f(t)$ (exact) = 0.72 G + 0.15 B')
plt.plot(t_dense, f_t_neural,'--r',linewidth=2, label='f(t) PINN', alpha=0.3)
plt.grid()
plt.title(f'Noise level: {noise}')
plt.xlabel('t')
plt.ylabel('$f(t)$')
plt.legend()
plt.savefig(f'./ft_noise_{noise}.png', dpi=300)



