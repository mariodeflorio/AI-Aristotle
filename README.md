
| Table of Contents |
| ----------------- |
| [1. Introduction(AI-Aristotle)](#Introduction) |
| [2. General Installation](#General-Installation) |
| [3. Getting Started](#Getting-Started) |
| [4. Contribution](#contribution) |
| [5. To Do](#To-Do) |
| [6. Contact](#Contact) |



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



# Getting Started

Installing the libraries:

- CSV


# Contribution
  
  
# To Do

  
# Contact
Email addresses: 
- Nazanin@Brown.edu 
- mario_de_florio@brown.edu
- khemraj_shukla@brown.edu
