# rl-xbandit-demo

## Run the bandit comparison

Use the project virtual environment on Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\bandit_experiment.py --arms 10 --runs 2000 --steps 1000
```

The script compares epsilon-greedy, optimistic initialization, and UCB on a
Gaussian N-armed testbed and saves a learning-curve plot to
`src\learning_curves.png`.
