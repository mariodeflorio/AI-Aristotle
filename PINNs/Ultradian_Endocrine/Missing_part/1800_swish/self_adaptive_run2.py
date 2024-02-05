
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import optax
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import time

#Defining the model
def glucose_insulin_model(
    t,
    meal_t,
    meal_q,
    Vp=3,
    Vi=11,
    Vg=10,
    E=0.2,
    tp=6,
    ti=100,
    td=12,
    k=1 / 120,
    Rm=209,
    a1=6.6,
    C1=300,
    C2=144,
    C3=100,
    C4=80,
    C5=26,
    Ub=72,
    U0=4,
    Um=90,
    Rg=180,
    alpha=7.5,
    beta=1.772,
):
    def func(y, t):
        f1 = Rm / (1 + np.exp(-y[2] / Vg / C1 + a1))
        f2 = Ub * (1 - np.exp(-y[2] / Vg / C2))
        kappa = (1 / Vi + 1 / E / ti) / C4
        f3 = (U0 + Um / (1 + (kappa * y[1]) ** (-beta))) / Vg / C3
        f4 = Rg / (1 + np.exp(alpha * (y[5] / Vp / C5 - 1)))
        IG = np.sum(
            meal_q * k * np.exp(k * (meal_t - t)) *
            np.heaviside(t - meal_t, 0.5)
        )
        tmp = E * (y[0] / Vp - y[1] / Vi)
        return [
            f1 - tmp - y[0] / tp,
            tmp - y[1] / ti,
            f4 + IG - f2 - f3 * y[2],
            (y[0] - y[3]) / td,
            (y[3] - y[4]) / td,
            (y[4] - y[5]) / td,
        ]

    # defining initial conditions for 6 state variables
    Vp0, Vi0, Vg0 = 3, 11, 10
    y0 = [12 * Vp0, 4 * Vi0, 110 * Vg0 ** 2, 0, 0, 0]
    return odeint(func, y0, t)

meal_t = np.array([300, 650, 1100, 2000])
meal_q = np.array([60e3, 40e3, 50e3, 100e3])

# meal_t = np.array([300, 650])
# meal_q = np.array([60e3, 40e3])


# t = np.arange(0, 1200, 1)[:, None]
# y = glucose_insulin_model(np.ravel(t), meal_t, meal_q)

t_dense = jnp.linspace(0, 1800, 1801)[:, None]
y_dense_org = glucose_insulin_model(np.ravel(t_dense),meal_t, meal_q)

scale_factor = [1,1,100,1,1,1]
y_dense = y_dense_org/scale_factor

sample_rate   = 1
# sample_rate_G = 3
t_data = t_dense[::sample_rate,0:1]
Ip_data = y_dense[::sample_rate,0:1]
Ii_data = y_dense[::sample_rate,1:2]
G_data = y_dense[::sample_rate,2:3]
h1_data = y_dense[::sample_rate,3:4]
h2_data = y_dense[::sample_rate,4:5]
h3_data = y_dense[::sample_rate,5:6]


data = jnp.concatenate([Ip_data, Ii_data, G_data, h1_data, h2_data, h3_data], axis=1)
t_i  = jnp.array([[0]])
IC   = jnp.concatenate([Ip_data[0:1,:], Ii_data[0:1,:], G_data[0:1,:], h1_data[0:1,:], h2_data[0:1,:], h3_data[0:1,:]], axis=1)
# IC   = jnp.concatenate([Ip_data[0:1,:], Ii_data[0:1,:], G_data[0:1,:], h1_data[0:1,:], h2_data[0:1,:], h3_data[0:1,:]], axis=1)
plt.figure()
plt.scatter(t_data, data[:,0:1], alpha=0.3)
plt.plot(t_dense,y_dense[:,0],'-r',label='Ip')

plt.figure()
plt.scatter(t_data, data[:,1:2], alpha=0.3)
plt.plot(t_dense,y_dense[:,1],'-b',label='Ii')

plt.figure()
plt.scatter(t_data, data[:,2:3], alpha=0.3,label='G_sampled')
plt.plot(t_dense,y_dense[:,2],'-g',label='G')
plt.legend()

Vp=3
Vi=11
Vg=10
E=0.2
tp=6
ti=100
td=12
Ip_true = y_dense[:, 0:1]*scale_factor[0]
Ii_true = y_dense[:, 1:2]*scale_factor[1]
f_t_true = -E * (Ip_true / Vp - Ii_true / Vi) - Ip_true/tp
g_t_true =  E * (Ip_true / Vp - Ii_true / Vi) - Ii_true/ti
extra= np.column_stack((f_t_true, g_t_true))

tmin, tmax = t_dense[0,0], t_dense[-1,0]

def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(key, shape=(n_in, n_out)) / jnp.sqrt(n_in) # random initialization
        B = jax.random.normal(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params


#_________________With feature layer______________________
def feature_transform(t):
    t = 0.01 * t
    return jnp.concatenate(
        (t, jnp.sin(t), jnp.sin(2 * t), jnp.sin(3 * t), jnp.sin(4 * t), jnp.sin(5 * t)),
        axis=1,
    )



def output_transform(t, y, y_dense, t_dense):
    idx = 1800
    k = (y_dense[idx] - y_dense[0]) / (t_dense[idx] - t_dense[0])
    b = (t_dense[idx] * y_dense[0] - t_dense[0] * y_dense[idx]) / (
            t_dense[idx] - t_dense[0])
    linear = k * t + b
    factor = 1  # jnp.tanh(t) * jnp.tanh(idx - t)

    return linear + factor * y

def fwd(params, t):
    X = feature_transform(t)  # Apply the feature_transform to input t
    inputs= X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.swish(inputs @ layer['W'] + layer['B'])

    Y= inputs @ last['W'] + last['B']
    Y= output_transform(t, Y, y_dense, t_dense)
    return Y



#__________________extra__________________________________________

def fwd_f(params, t):
    X = feature_transform(t)  # Apply the feature_transform to input t
    # X =  0.01 * t
    inputs= X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.swish(inputs @ layer['W'] + layer['B'])

    Y= inputs @ last['W'] + last['B']
    Y= output_transform(t, Y, extra, t_dense)
    return Y

def fwd_g(params, t):
    X = feature_transform(t)  # Apply the feature_transform to input t
    # X =  0.01 * t
    inputs= X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.swish(inputs @ layer['W'] + layer['B'])

    Y= inputs @ last['W'] + last['B']
    Y= output_transform(t, Y, g_t_true, t_dense)
    return Y



@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)

#@title **Inverse**
def ODE_loss(t, y1, y2, y3, y4, y5, y6, f_t, g_t):
    Vp=3
    Vi=11
    Vg=10
    E=0.2
    tp=6
    ti=100
    td=12
    k=1 / 120
    Rm=209
    a1=6.6
    C1=300
    C2=144
    C3=100
    C4=80
    C5=26
    Ub=72
    U0=4
    Um=90
    Rg=180
    alpha=7.5
    beta=1.772


    Ip = y1(t)*scale_factor[0]
    Ii = y2(t)*scale_factor[1]
    G  = y3(t)*scale_factor[2]
    h1 = y4(t)*scale_factor[3]
    h2 = y5(t)*scale_factor[4]
    h3 = y6(t)*scale_factor[5]

    meal_t = np.array([300, 650, 1100, 2000])
    meal_q = np.array([60e3, 40e3, 50e3, 100e3])

    # meal_t = np.array([300, 650])
    # meal_q = np.array([60e3, 40e3])

    f1 = Rm * jax.nn.sigmoid(G / (Vg * C1) - a1)
    f2 = Ub * (1 - jnp.exp(-G / (Vg * C2)))
    kappa = (1 / Vi + 1 / (E * ti)) / C4
    f3 = (U0 + Um / (1 + jnp.power(jnp.maximum(kappa * Ii, 1e-3), -beta))) / (Vg * C3)
    f4 = Rg * jax.nn.sigmoid(alpha * (1 - h3 / (Vp * C5)))

    dt = t - meal_t
    IG = jnp.sum(0.5 * meal_q * k * jnp.exp(-k * dt) * (jnp.sign(dt) + 1), axis=1, keepdims=True)


    y1_t = lambda t: jax.grad(lambda t: jnp.sum(y1(t)))(t)
    y2_t = lambda t: jax.grad(lambda t: jnp.sum(y2(t)))(t)
    y3_t = lambda t: jax.grad(lambda t: jnp.sum(y3(t)))(t)
    y4_t = lambda t: jax.grad(lambda t: jnp.sum(y4(t)))(t)
    y5_t = lambda t: jax.grad(lambda t: jnp.sum(y5(t)))(t)
    y6_t = lambda t: jax.grad(lambda t: jnp.sum(y6(t)))(t)


    ode1 = y1_t(t)*scale_factor[0] - (f1 + f_t(t))
    ode2 = y2_t(t)*scale_factor[1] - (g_t(t))
    ode3 = y3_t(t) - 1/scale_factor[2]*(f4 + IG - f2 - f3 * G)
    ode4 = y4_t(t)*scale_factor[3] - (Ip - h1) / td
    ode5 = y5_t(t)*scale_factor[4] - (h1 - h2) / td
    ode6 = y6_t(t)*scale_factor[5] - (h2 - h3) / td

    return ode1, ode2, ode3, ode4, ode5, ode6

N_c = 1800
t_c = jnp.linspace(tmin, tmax, N_c+1)[:, None]
def loss_fun(params, params_f, params_g, l1, l2, l3, l4, l5, l6, t_i, t_d, t_c, data_IC, data):

    y1_func = lambda t: fwd(params, t)[:, [0]]
    y2_func = lambda t: fwd(params, t)[:, [1]]
    y3_func = lambda t: fwd(params, t)[:, [2]]
    y4_func = lambda t: fwd(params, t)[:, [3]]
    y5_func = lambda t: fwd(params, t)[:, [4]]
    y6_func = lambda t: fwd(params, t)[:, [5]]

    # f_t     = lambda t: fwd_f(params_f, t)[:, [0]]
    # g_t     = lambda t: fwd_g(params_g, t)[:, [0]]
    f_t     = lambda t: fwd_f(params_f, t)[:, [0]]
    g_t     = lambda t: fwd_f(params_f, t)[:, [1]]

    loss_y1, loss_y2, loss_y3, loss_y4, loss_y5, loss_y6 = ODE_loss(t_c, y1_func, y2_func, y3_func, y4_func, y5_func, y6_func, f_t, g_t)

    loss_y1 = l1*loss_y1
    loss_y2 = l2*loss_y2
    loss_y3 = l3*loss_y3
    loss_y4 = l4*loss_y4
    loss_y5 = l5*loss_y5
    loss_y6 = l6*loss_y6

    loss_ode1 = jnp.mean(loss_y1 ** 2)
    loss_ode2 = jnp.mean(loss_y2 ** 2)
    loss_ode3 = jnp.mean(loss_y3 ** 2)
    loss_ode4 = jnp.mean(loss_y4 ** 2)
    loss_ode5 = jnp.mean(loss_y5 ** 2)
    loss_ode6 = jnp.mean(loss_y6 ** 2)


    # Compute the loss for the initial conditions
    t_i = t_i.flatten()[:,None]
    pred_IC = jnp.concatenate([y1_func(t_i), y2_func(t_i), y3_func(t_i), y4_func(t_i), y5_func(t_i), y6_func(t_i)],axis=1)
    loss_IC = MSE(data_IC, pred_IC)
    # Compute the loss for Y_data
    t_d1    = t_d[::1].flatten()[:,None]
    pred_d = jnp.concatenate([y1_func(t_d1), y2_func(t_d1), y3_func(t_d1), y4_func(t_d1), y5_func(t_d1), y6_func(t_d1)],axis=1)

    t_d2    = t_d[::1].flatten()[:,None]
    pred_d2 = jnp.concatenate([y1_func(t_d2), y2_func(t_d2), y3_func(t_d2), y4_func(t_d2), y5_func(t_d2), y6_func(t_d2)],axis=1)

    loss_data = MSE(data[::1,0:1], pred_d[:,0:1])
    loss_data += MSE(data[::1,2:3], pred_d2[:,2:3])
    # loss_data = MSE(data[:, [0, 2]], pred_d[:, [0, 2]])


    return loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3, loss_ode4, loss_ode5, loss_ode6


def loss_fun_total(params, params_f, params_g, l1,l2, l3, l4, l5, l6, t_i, t_d, t_c, data_IC, data, loss_weight):

    loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3, loss_ode4, loss_ode5, loss_ode6= loss_fun(params, params_f, params_g, l1, l2, l3, l4, l5, l6, t_i, t_d, t_c, data_IC, data)


    loss_total = loss_weight[0]*loss_IC+ loss_weight[1]*loss_data\
                + loss_weight[2]*loss_ode1+ loss_weight[3]*loss_ode2\
                + loss_weight[4]*loss_ode3 + loss_weight[5]*loss_ode4 + loss_weight[6]*loss_ode5 + loss_weight[7]*loss_ode6

    return loss_total


def loss_fun_total_lbfgs(params_total, t_i, t_d, t_c, data_IC, data, loss_weight):
    params, params_f, params_g, l1,l2, l3, l4, l5, l6 = \
    params_total[0], params_total[1], params_total[2], params_total[3], \
    params_total[4], params_total[5], params_total[6], params_total[7], \
    params_total[8]

    loss_IC, loss_data, loss_ode1, loss_ode2, loss_ode3, loss_ode4, loss_ode5, loss_ode6= loss_fun(params, params_f, params_g, l1, l2, l3, l4, l5, l6, t_i, t_d, t_c, data_IC, data)


    loss_total = loss_weight[0]*loss_IC+ loss_weight[1]*loss_data\
                + loss_weight[2]*loss_ode1+ loss_weight[3]*loss_ode2\
                + loss_weight[4]*loss_ode3 + loss_weight[5]*loss_ode4 + loss_weight[6]*loss_ode5 + loss_weight[7]*loss_ode6
    print(f"Loss: {loss_total}")

    return loss_total


@jax.jit
def update(opt_state, opt_state_f, opt_state_g, opt_state_l1,opt_state_l2, opt_state_l3,\
           opt_state_l4, opt_state_l5, opt_state_l6, params, params_f, params_g,\
           params_l1, params_l2, params_l3, params_l4, params_l5, params_l6, t_i,\
           t_data, t_c, IC, data, loss_weight\
           ):
  grads=jax.grad(loss_fun_total, argnums=[0,1,2,3,4,5,6,7,8])(params, params_f, params_g, params_l1, params_l2, params_l3, params_l4, params_l5, params_l6, t_i, t_data, t_c, IC, data, loss_weight)

  #Update params
  updates, opt_state = optimizer.update(grads[0], opt_state)
  params = optax.apply_updates(params, updates)

  updates_f, opt_state_f = optimizer.update(grads[1], opt_state_f)
  params_f = optax.apply_updates(params_f, updates_f)

  updates_g, opt_state_g = optimizer.update(grads[2], opt_state_g)
  params_g = optax.apply_updates(params_g, updates_g)

  updates_l1, opt_state_l1 = optimizer.update(-grads[3], opt_state_l1)
  params_l1 = optax.apply_updates(params_l1, updates_l1)

  updates_l2, opt_state_l2 = optimizer.update(-grads[4], opt_state_l2)
  params_l2 = optax.apply_updates(params_l2, updates_l2)

  updates_l3, opt_state_l3 = optimizer.update(-grads[5], opt_state_l3)
  params_l3 = optax.apply_updates(params_l3, updates_l3)

  updates_l4, opt_state_l4 = optimizer.update(-grads[6], opt_state_l4)
  params_l4 = optax.apply_updates(params_l4, updates_l4)

  updates_l5, opt_state_l5 = optimizer.update(-grads[7], opt_state_l5)
  params_l5 = optax.apply_updates(params_l5, updates_l5)

  updates_l6, opt_state_l6 = optimizer.update(-grads[8], opt_state_l6)
  params_l6 = optax.apply_updates(params_l6, updates_l6)

  # updates_l1, opt_state_l1 = optimizer.update(-grads[9], opt_state_l1)
  # params_l1 = optax.apply_updates(params_l1, updates_l1)

  return opt_state, params, opt_state_f, params_f, opt_state_g, params_g,\
   opt_state_l1, opt_state_l2, opt_state_l3, opt_state_l4, opt_state_l5,\
    opt_state_l6, params_l1, params_l2, params_l3, params_l4, params_l5, params_l6

params = init_params([6] + [128]*3+[6])
params_f = init_params([6] + [32]*3+[2])
params_g = init_params([6] + [32]*3+[1])

keys = jax.random.split(jax.random.PRNGKey(0), 10)

lambda_1 = jax.random.uniform(keys[0], shape=(N_c + 1, 1))
lambda_2 = jax.random.uniform(keys[1], shape=(N_c + 1, 1))
lambda_3 = jax.random.uniform(keys[2], shape=(N_c + 1, 1))
lambda_4 = jax.random.uniform(keys[3], shape=(N_c + 1, 1))
lambda_5 = jax.random.uniform(keys[4], shape=(N_c + 1 , 1))
lambda_6 = jax.random.uniform(keys[5], shape=(N_c + 1, 1))


optimizer = optax.adam(1e-3)

opt_state = optimizer.init(params)
opt_state_f = optimizer.init(params_f)
opt_state_g = optimizer.init(params_g)

opt_state_l1 = optimizer.init(lambda_1)
opt_state_l2 = optimizer.init(lambda_2)
opt_state_l3 = optimizer.init(lambda_3)
opt_state_l4 = optimizer.init(lambda_4)
opt_state_l5 = optimizer.init(lambda_5)
opt_state_l6 = optimizer.init(lambda_6)


# Training parameters
epochs_phase1 = 10000 #10000
epochs_phase2 = 1000000 #10000
loss_weight_phase1 = [1, 1, 0, 0, 0, 0, 0, 0]
loss_weight_phase2 = [1, 1, 1, 1, 1, 1, 1, 1]

loss_his, loss_indi_his, epoch_his = [], [], []

start_time = time.time()
# Training loop - Two-step training
for ep in range(epochs_phase1 + epochs_phase2 + 1):
    if ep <= epochs_phase1:
        loss_weight = loss_weight_phase1
    else:
        loss_weight = loss_weight_phase2

    opt_state, params, opt_state_f, params_f, opt_state_g, params_g,\
    opt_state_l1, opt_state_l2, opt_state_l3, opt_state_l4, opt_state_l5,\
    opt_state_l6, params_l1, params_l2, params_l3, params_l4, params_l5, params_l6 = update(opt_state, opt_state_f, opt_state_g,
                                                                                          opt_state_l1, opt_state_l2, opt_state_l3,
                                                                                          opt_state_l4, opt_state_l5, opt_state_l6,
                                                                                          params, params_f, params_g,
                                                                                          lambda_1, lambda_2, lambda_3,
                                                                                          lambda_4, lambda_5, lambda_6, t_i,
                                                                                          t_data, t_c, IC, data, loss_weight)


    if ep %(1000) ==0:
      loss_val = loss_fun_total(params, params_f, params_g , lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, t_i, t_data, t_c, IC, data, loss_weight)
      loss_val_individual = loss_fun(params, params_f, params_g, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, t_i, t_data, t_c, IC, data)
      epoch_his.append(ep)
      loss_his.append(loss_val)
      loss_indi_his.append(loss_val_individual)
      print(f'Epoch={ep}, \t loss={loss_val:.2e}, \t loss_IC={loss_val_individual[0]:.2e},\
       \t loss_d={loss_val_individual[1]:.2e}, \t loss_e1={loss_val_individual[2]:.2e},\
        \t loss_e2={loss_val_individual[3]:.2e}, \t loss_e3={loss_val_individual[4]:.2e},\
         \t loss_e4={loss_val_individual[5]:.2e}, \t loss_e5={loss_val_individual[6]:.2e},\
          \t loss_e6={loss_val_individual[7]:.2e}')
      end_time = time.time()
        
      running_time = end_time - start_time
      print(f"Total running time: {running_time:.4f} seconds")



np.savez('./params_swish_5_2_1800.npz', *params)
np.savez('./params_f_swish_5_2_1800.npz', *params_f)
# np.savez('/content/drive/MyDrive/params_l1_swish_5_5_1800.npz', *params_l1)
# np.savez('/content/drive/MyDrive/params_l2_swish_5_5_1800.npz', *params_l2)
# np.savez('/content/drive/MyDrive/params_l3_swish_5_5_1800.npz', *params_l3)
# np.savez('/content/drive/MyDrive/params_l4_swish_5_5_1800.npz', *params_l4)
# np.savez('/content/drive/MyDrive/params_l5_swish_5_5_1800.npz', *params_l5)
# np.savez('/content/drive/MyDrive/params_l6_swish_5_5_1800.npz', *params_l6)
np.savez(f'./loss_data_epoch_swish_5_2_1800.npz', epoch_his=epoch_his, loss_his=loss_his, loss_indi_his=loss_indi_his)



SAVE_FIG = True
#History of loss
font = 12
w_size, h_size = 8, 4.5
save_results_to='./'

fig, ax = plt.subplots()
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=6)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
plt.xlabel('$iteration$', fontsize = font)
plt.ylabel('$loss values$', fontsize = font)
plt.yscale('log')
plt.grid(True)
plt.plot(epoch_his, np.asarray(loss_his), label='$total loss $')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,0], label='$loss IC$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,1], label='$loss data$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,2], label='$loss eqn1$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,3], label='$loss eqn2$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,4], label='$loss eqn3$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,5], label='$loss eqn4$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,6], label='$loss eqn5$')
plt.plot(epoch_his, np.asarray(loss_indi_his)[:,7], label='$loss eqn6$')
plt.legend(loc="upper right",  fontsize = 8, ncol=4)
# plt.legend()
ax.tick_params(axis='both', labelsize = 12)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'History_loss.png', dpi=300)

pred = fwd(params,t_dense)


font = 12
labelsize =12
w_size, h_size = 8, 4.5

fig, ax = plt.subplots()
ax.plot(t_dense, y_dense[:, 0:1]*scale_factor[0],'-g',label='$I_p$ (true)')
ax.plot(t_dense, pred[:,0]*scale_factor[0],'--k',linewidth=2, label='SBINN')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.legend(fontsize=font)
ax.tick_params(axis='both', labelsize = labelsize)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax.grid(True)
ax.set_xlabel('time', fontsize = font)
ax.set_ylabel('$I_p$', fontsize = font)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'Ip.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(t_dense, y_dense[:, 1:2]*scale_factor[1],'-g',label='$I_i$ (true)')
ax.plot(t_dense, pred[:,1]*scale_factor[1],'--k',linewidth=2, label='SBINN')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.legend(fontsize=font)
ax.tick_params(axis='both', labelsize = labelsize)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax.grid(True)
ax.set_xlabel('time', fontsize = font)
ax.set_ylabel('$I_i$', fontsize = font)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'Ii.png', dpi=300)


fig, ax = plt.subplots()
ax.plot(t_dense, y_dense[:, 2:3]*scale_factor[2],'-g',label='G (true)')
ax.plot(t_dense, pred[:,2]*scale_factor[2],'--k',linewidth=2, label='SBINN')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.legend(fontsize=font)
ax.tick_params(axis='both', labelsize = labelsize)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax.grid(True)
ax.set_xlabel('time', fontsize = font)
ax.set_ylabel('$G$', fontsize = font)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'G.png', dpi=300)

Vp=3
Vi=11
Vg=10
E=0.2
tp=6
ti=100
td=12
Ip_true = y_dense[:, 0:1]*scale_factor[0]
Ii_true = y_dense[:, 1:2]*scale_factor[1]
f_t_true = -E * (Ip_true / Vp - Ii_true / Vi) - Ip_true/tp
g_t_true =  E * (Ip_true / Vp - Ii_true / Vi) - Ii_true/ti


Ip_pred = pred[:,0:1]*scale_factor[0]
Ii_pred = pred[:,1:2]*scale_factor[1]
G_pred = pred[:,2:3]*scale_factor[2]

f_t_pred = fwd_f(params_f, t_dense)[:,0]
g_t_pred = fwd_f(params_f, t_dense)[:,1]

df_f = pd.DataFrame({"t": np.ravel(t_dense),"Ip": np.ravel(Ip_pred), "Ii": np.ravel(Ii_pred), "ft": np.ravel(f_t_pred) })
df_f.to_csv("./ft_1800.csv", index=False)

df_pred = pd.DataFrame({"t": np.ravel(t_dense),"Ip": np.ravel(Ip_pred), "Ii": np.ravel(Ii_pred), "G": np.ravel(G_pred) })
df_pred.to_csv("./pred_final.csv", index=False)

df_g = pd.DataFrame({"t": np.ravel(t_dense),"Ip": np.ravel(Ip_pred), "Ii": np.ravel(Ii_pred), "gt": np.ravel(g_t_pred) })
df_g.to_csv("./gt_1800.csv", index=False)



font = 12
labelsize =12
w_size, h_size = 8, 4.5

fig, ax = plt.subplots()
ax.plot(t_dense, f_t_true, label='$f_t$ (true)')
ax.plot(t_dense, f_t_pred,'--k',linewidth=2, label='SBINN')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.legend(fontsize=font)
ax.tick_params(axis='both', labelsize = labelsize)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax.grid(True)
ax.set_xlabel('time', fontsize = font)
ax.set_ylabel('$f_t$', fontsize = font)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'ft.png', dpi=300)


fig, ax = plt.subplots()
ax.plot(t_dense, g_t_true, label='$g_t$ (true)')
ax.plot(t_dense, g_t_pred,'--k',linewidth=2, label='SBINN')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.legend(fontsize=font)
ax.tick_params(axis='both', labelsize = labelsize)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax.grid(True)
ax.set_xlabel('time', fontsize = font)
ax.set_ylabel('$g_t$', fontsize = font)
fig.set_size_inches(w=w_size,h=h_size)
if SAVE_FIG:
    plt.savefig(save_results_to +'gt.png', dpi=300)
