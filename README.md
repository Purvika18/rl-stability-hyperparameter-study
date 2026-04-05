# RL Stability & Hyperparameter Study 🤖

A comparative study of REINFORCE and PPO on the CartPole-v1 
environment. The goal wasn't just to get good scores — it was 
to understand how sensitive each algorithm is to hyperparameter 
choices and how stable their training actually is across runs.

Done as a seminar paper for my AI course at Hochschule Schmalkalden.

---

## What I looked at

**Algorithms compared:**
- REINFORCE (vanilla policy gradient)
- PPO (Proximal Policy Optimization)

**Hyperparameters tested:**
- Gamma (discount factor) sensitivity
- Learning rate sensitivity  
- Clip range sensitivity (PPO only)

**Evaluation approach:**
- Multi-seed experiments to separate luck from actual performance
- Mean and standard deviation across runs to measure stability
- Not just "which one wins" but "which one is more reliable"

---

## Key findings

- PPO significantly more stable than REINFORCE across seeds
- REINFORCE highly sensitive to learning rate — small changes 
  cause large performance swings
- PPO's clip range has a sweet spot — too small or too large 
  both hurt
- Gamma sensitivity differs between algorithms in interesting ways

---

## How to run it
```bash
# Install dependencies
pip install stable-baselines3 gymnasium matplotlib numpy

# Run experiments
python train.py

# Plot results
python plot_results.py
```

---

## Built with

Python · Stable-Baselines3 · Gymnasium · Matplotlib · NumPy

---

## Why this matters

Hyperparameter sensitivity is one of the most underappreciated 
problems in RL. A method that works brilliantly with one set of 
parameters can completely fail with slightly different ones. 
Understanding this is more useful than just running the default 
settings and reporting the score.
