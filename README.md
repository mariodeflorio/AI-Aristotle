

# AI-Aristotle framework for gray-box identification
![AI Aristotle Diagram](https://github.com/mariodeflorio/AI-Aristotle/blob/main/AI_aristotle_diagram.jpg?raw=true)
1. The observed data and the partial knowledge of physics are used to train the selected neural network-based module.
2. The selection of the neural networks-based module needs to be done between (a) X-TFC, recommended for high-resolution
data and missing terms discovery, and (b) PINN, recommended for sparse data and parameter estimation. The neural
network outputs are the time-dependent representations of the missing terms of the dynamical systems, which are fed into the
symbolic regression algorithm.
3. The selected Symbolic Regression module identifies the mathematical expressions of the missing terms. It is recommended
to use both symbolic regressors for cross-validation.
4. The full knowledge of physics is now available, allowing forward modeling performance


| Table of Contents |
| ----------------- |
| [1. Introduction (AI-Aristotle)](#Introduction) |
| [2. General Installation](#General-Installation) |
| [3. Getting Started](#Getting-Started) |
| [4. Contributions](#contributions) |
| [5. Contact](#Contact) |


# Introduction

The source code for the manuscript: AI-Aristotle: A Physics-Informed framework for Systems Biology Gray-Box Identification

Discovering mathematical equations that govern physical and biological systems from observed data is a fundamental challenge in scientific research. We present a new physics-informed framework for parameter estimation and missing physics identification (gray-box) in the field of Systems Biology. The proposed framework -- named AI-Aristotle -- combines eXtreme Theory of Functional Connections (X-TFC) domain-decomposition and Physics-Informed Neural Networks (PINNs) with symbolic regression (SR) techniques for parameter discovery and gray-box identification. We test the accuracy, speed, flexibility and robustness of AI-Aristotle based on two benchmark problems in Systems Biology: a pharmacokinetics drug absorption model, and an ultradian endocrine model for glucose-insulin interactions. We compare the two machine learning methods (X-TFC and PINNs), and moreover, we employ two different symbolic regression techniques to cross-verify our results. While the current work focuses on the performance of AI-Aristotle based on synthetic data, it can equally handle noisy experimental data and can even be used for black-box identification in just a few minutes on a laptop. More broadly, our work provides insights into the accuracy, cost, scalability, and robustness of integrating neural networks with symbolic regressors, offering a comprehensive guide for researchers tackling gray-box identification challenges in complex dynamical systems in biomedicine and beyond.





# General Installation

## PINNs General Installation Instructions

To set up your environment for this project, ensure you have Python installed on your system. If not, download it from  [Python website](https://www.python.org/). This project is compatible with Python 3.x versions.

### Install Required Libraries

Open your terminal or command prompt and execute the following commands to install the necessary libraries:

```bash
# Upgrade pip and install JAX (For specific CPU or GPU support, refer to JAX's installation guide)
pip install --upgrade pip
pip install --upgrade jax jaxlib

# Install NumPy
pip install numpy

# Install Matplotlib for plotting
pip install matplotlib

# Install Optax for optimization
pip install optax

# Install Pandas for data manipulation
pip install pandas

# Install SciPy for scientific computations
pip install scipy

# Check for any missing dependencies
pip check
```

## X-TFC

Install MATLAB R2023b.


# Getting Started

## PINNs
After installing all the required libraries, you will be able to run the `.py` files included in this repository and obtain results. Ensure you follow the installation instructions carefully to set up your environment correctly.

### How to Adapt the Code for Your Problem

#### Modifying the ODE System and Initial Conditions

If your problem involves a different number of ordinary differential equations (ODEs) or you need to change the initial conditions, you can adjust the loss function and initial conditions accordingly. Here's how you can do it:

1. **Edit the Loss Function**: Depending on the number of ODEs your problem has, modify the loss function in the code to reflect this change.

2. **Change Initial Conditions (IC)**: Update the initial conditions to match those of your specific problem. You can do this by altering the values where the initial conditions are defined in the code.

#### Importing and Using Your Data

To use your own data, either synthetic or real, replace the example file path with the path to your data file. Here is an example of how to read and prepare your data:

```python
file_path_1 = './your_data_file.csv'  # Replace with your actual file path
data_noisy = pd.read_csv(file_path_1)
t_data = data['t']
t_data = t_data.to_numpy(dtype=np.float32).reshape(-1, 1)
data = jnp.column_stack((data['Variable1'], data['Variable2'], data['Variable3']))
```

## X-TFC


### X-TFC for parameter discovery

To perform parameter discovery with X-TFC for the drugs absorption compartmental model, open and run the Matlab file *PK_parameter_discovery.m*. The script loads the selected synthetic dataset. For example, if the user selects *drug_real_10.csv*, the dataset with 10 data points will be used in the simulation. The user can add a noise percentage to it by modifying the variable *noise_lev*. Follows the list of tunable parameters:

- *N*: number of collocation points per each sub-domain
- *m*: number of neurons
- *t_step*: length of sub-domains
- *final_subdomain*: (n_t - 1) to consider the whole time domain or 1 to consider only the first sub-domain.
- *LB*: Lower boundary for weight and bias samplings
- *UB*: Upper boundary for weight and bias samplings
- *IterMax*: maximum number of iterations of the least-squares algorithm 
- *IterTol*: tolerance of the least-squares algorithm
- *type_act*: select the activation function to use.

The script prints the computational time of the total execution, the values of the discovered parameters, and their relative errors compared to the exact values.

### X-TFC for missing term discovery

To perform missing term discovery with X-TFC for the drugs absorption compartmental model, open and run the Matlab file *PK_missing_term.m*. The script loads the selected synthetic dataset. For example, if the user selects *drug_real_10.csv*, the dataset with 10 data points will be used in the simulation. The user can add a noise percentage to it by modifying the variable *noise_lev*. Follows the list of tunable parameters:

- *N*: number of collocation points per each sub-domain
- *m*: number of neurons
- *t_step*: length of sub-domains
- *N_test*: number of test points.
- *LB*: Lower boundary for weight and bias samplings
- *UB*: Upper boundary for weight and bias samplings
- *IterMax*: maximum number of iterations of the least-squares algorithm 
- *IterTol*: tolerance of the least-squares algorithm
- *type_act*: select the activation function to use.

The script prints the computational time of the total execution, the values of the Mean Absolute Error, Mean Squared Error, and Relative Error of the discovered term compared to the exact term. Two figures will be produced, in which the solutions of the differential equations and the missing term are plotted vs. exact solutions for both training and test points. The vectors of the time domain, the three solutions of the differential equations, and the discovered term are saved in the file *t_B_G_U_f.csv*, which will be used in the Symbolic Regression algorithm for the mathematical distillation.


# Contributions

This project encompasses various components, each spearheaded by different contributors. Should you encounter any issues or have questions about a specific part of the project, please reach out to the respective contributor for assistance:

- **Physics-Informed Neural Networks (PINNs)**: For inquiries or issues related to the PINNs code, please contact Nazanin. She is the author and responsible for this segment and can provide the necessary support.

- **X-TFC**: If your questions or issues pertain to the X-TFC component, Mario is your point of contact. He has developed this part and will be able to assist you with any related queries.

- **Symbolic Regression**: For matters concerning the Symbolic Regression part of the project, please get in touch with Khemraj. He is the main contributor for this section and will address any problems you might face.

Each contributor is responsible for their respective sections and is best equipped to provide support and answers for their part of the code.
 
  



  
# Contact
Email addresses: 
- Nazanin@brown.edu 
- mario_de_florio@brown.edu
- khemraj_shukla@brown.edu
