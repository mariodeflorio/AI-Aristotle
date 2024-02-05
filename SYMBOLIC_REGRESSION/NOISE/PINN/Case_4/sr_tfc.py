import pysr
import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
import csv
import sys
import matplotlib.pyplot as plt

## Data Import
G = []
B = []
f = []
t = []


weights = (1 /np.sqrt( 0.04)) * np.ones((501,))


with open('pred_500_0.04.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      G.append(float(row[1]))
      B.append(float(row[2]))
      f.append(float(row[4]))
      t.append(float(row[0]))
      line_count += 1
    print(f'Processed {line_count} lines.')

G = np.array(G).reshape(-1,1)
B = np.array(B).reshape(-1,1)
f = np.array(f).reshape(-1,1)
t = np.array(t).reshape(-1,1)
d = np.concatenate((G, B, f), axis=1)
y = d[:, 2]
X = d[:, 0:2]


pysr_params = dict(
    populations=50,
    procs = 8,
    #ncyclesperiteration=2,
    population_size=30,
    model_selection="best",
    loss="myloss(x, y, w) = w * abs(x - y)",
    binary_operators=["plus", "sub", "mult"],
    maxsize=9,               
)

model = PySRRegressor(
    niterations=50,
    **pysr_params
)

# Run model:
#model = PySRRegressor.from_file("hall_of_fame_2024-01-17_155335.906.pkl")
model.fit(X, y, weights=weights)
y_pred = model.predict(X)
err = np.square(np.subtract(y,y_pred)).mean()

best_idx = model.equations_.query(
    f"loss < {2 * model.equations_.loss.min()}"
).score.idxmax()


print(f"Best model:", model.sympy(best_idx))

print(f"Projection Error: {err}")
print(f"Model: {model}")

print(f"Model SymPy: {model.sympy()}")

#print("4 model:", model.sympy(5))
print(f"Latex Equation: {model.latex()}")
