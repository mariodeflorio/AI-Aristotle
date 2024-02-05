import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import optax
from scipy.integrate import odeint
import pandas as pd
import time

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

sample_rate = 50
dp=  500/sample_rate

t_data = t_dense[::sample_rate,0:1]
G_data = y_dense[::sample_rate,0:1]
B_data = y_dense[::sample_rate,1:2]
U_data = y_dense[::sample_rate,2:3]

data = jnp.concatenate([G_data, B_data, U_data], axis=1)
t_i  = jnp.array([[0]])
IC   = jnp.concatenate([G_data[0:1,:], B_data[0:1,:], U_data[0:1,:]], axis=1)

# tmin,tmax=0.
tmin, tmax = t_dense[0,0], t_dense[-1,0]

#t = (t-np.min(t))/(np.max(t)-np.min(t))
def init_params(layers, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(key, shape=(n_in, n_out)) / jnp.sqrt(n_in) # random initialization
        B = jax.random.normal(key, shape=(n_out,))
        params.append({'W': W, 'B': B , 'kb': 0.5 , 'kg': 0.5})
    return params


def fwd(params,t):
  X = jnp.concatenate([t],axis=1)
  *hidden,last = params
  for layer in hidden :
    X = jax.nn.tanh(X@layer['W']+layer['B'])
  return X@last['W'] + last['B']

@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)



def ODE_loss(t, params, y1, y2, y3):
    kg = params[0]['kg']
    kb = params[0]['kb']
    G0 = 0.1
    # f_t=kg * y1(t) + kb * y2(t)

    y1_t = lambda t: jax.grad(lambda t: jnp.sum(y1(t)))(t)
    y2_t = lambda t: jax.grad(lambda t: jnp.sum(y2(t)))(t)
    y3_t = lambda t: jax.grad(lambda t: jnp.sum(y3(t)))(t)
    ode1 = y1_t(t) + kg * y1(t)
    ode2 = y2_t(t) - kg * y1(t) + kb * y2(t)
    ode3 = y3_t(t) - kb * y2(t)

    return ode1, ode2, ode3




#collocation points
N_c = 500

t_c = jnp.linspace(tmin, tmax, N_c+1)[:, None]
def loss_fun(params, t_i, t_d, t_c, data_IC, data):

    y1_func = lambda t: fwd(params, t)[:, [0]]
    y2_func = lambda t: fwd(params, t)[:, [1]]
    y3_func = lambda t: fwd(params, t)[:, [2]]
    # f_t     = lambda t: fwd(params_extra, t)[:, [0]]

    loss_y1, loss_y2, loss_y3 = ODE_loss(t_c, params, y1_func, y2_func, y3_func)

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


    return loss_IC+ loss_data+ loss_ode1+ loss_ode2+ loss_ode3



@jax.jit
def update(opt_state, params, t_i, t_data, t_c, IC, data):
  grads=jax.grad(loss_fun)(params, t_i, t_data, t_c, IC, data)

  #Update params
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return opt_state,params


########################################################################
optimizer = optax.adam(1e-4)
def run_simulation(seed):

  params = init_params([1] + [20]*4+[3], seed)
  optimizer = optax.adam(1e-4)
  opt_state = optimizer.init(params)

  # Initialize lists to store results
  kb_values_list = []
  kg_values_list = []
  epoch_numbers = []
  loss_values = []
  epochs = 50000

  print(f'Running with {dp} data points')
  start_time = time.time()
  for ep in range(epochs):
      opt_state, params = update(opt_state, params, t_i, t_data, t_c, IC, data)

      # Print loss and epoch info
      if ep % 1000 == 0:
          loss = loss_fun(params, t_i, t_data, t_c, IC, data)
          loss_values.append(loss)
          kb_updated = params[0]['kb']
          kg_updated = params[0]['kg']
          print(f'Epoch={ep}\tloss={loss:.3e} \t kb= {kb_updated} \t kg={kg_updated}')

          # Append the values directly to the lists
          kb_values_list.append(kb_updated)
          kg_values_list.append(kg_updated)
          epoch_numbers.append(ep)  # Store the current epoch number

          end_time = time.time()
          running_time = end_time - start_time
          print(f"Total running time: {running_time:.4f} seconds")
          

  np.savez(f'./params/params_{dp}_{i}.npz', *params)
  loss_values = np.array(loss_values)


  # Combine 'epoch_numbers', 'kb_values_list', and 'kg_values_list' into a single array
  results = np.column_stack((epoch_numbers, kb_values_list, kg_values_list))

  # Create a DataFrame from the combined results array
  df = pd.DataFrame(results, columns=['Epoch', 'kb', 'kg'])

  # Save the DataFrame to a CSV file
  df.to_csv(f'./params/drug_params_{dp}_{i}.csv', index=False)



  return kb_updated, kg_updated

final_kb_values = []
final_kg_values = []
used_seeds = []
#################### Run the simulation 10 times with different seeds ##################
for i in range(10):
    seed = np.random.randint(0, 10000)
    used_seeds.append(seed)
    print(f'################Run the simulation {i} time with different seed: {seed} ##################')
    kb, kg = run_simulation(seed)
    #final_kb_values.append(jnp.linalg.norm(0.15 - kb) / jnp.linalg.norm(0.15))
    #final_kg_values.append(jnp.linalg.norm(0.72 - kg) / jnp.linalg.norm(0.72))
    final_kb_values.append(np.abs(0.15 - kb))
    final_kg_values.append(np.abs(0.72 - kg))

# Calculate and print the average
average_kb = np.mean(final_kb_values)
std_kb = np.std(final_kb_values)
average_kg = np.mean(final_kg_values)
std_kg = np.std(final_kg_values)

print(f'Average kb error: {average_kb}')
print(f'Average kg error: {average_kg}')

