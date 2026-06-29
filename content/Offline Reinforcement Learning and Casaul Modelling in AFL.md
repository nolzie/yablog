---
title: Loose Ideas About Offline Reinforcement Learning in AFL
draft: false
tags:
  - mathematics
  - sport analytics
  - machine learning
---
  
In Australian Rules Football, we are drowning in "priors." We carry a century's worth of traditional lore, expert commentary, and unwritten rules passed down through generations of coaches. We know a long kick down the line to a contest is "safe." We know winning a clearance from a stoppage is inherently positive.

But when it comes to truly modeling the AFL, these priors often get in the way.

When we analyze the game with baked-in assumptions, we accidentally bias our data. We naturally design metrics that reward actions because they look like traditional good football, rather than because they objectively maximize scoring efficiency. To find a true "Moneyball" edge in the modern AFL, we have to strip away the human baggage and look at the game with entirely fresh eyes.


---

## 1. The Environment as a Markov Decision Process (MDP)

Traditional sports analytics methods treat invasion games as sequences of isolated, independent events. In contrast, I propose modelling football as a finite, episodic, multi-dimensional Markov Decision Process (MDP).

### 1.1 The State Space ($S$)
A state $s \in S$ represents a comprehensive snapshot of the spatial and tactical context of the game at any time $t$. Rather than treating space as continuous pixels, the field is discretized into a coordinate system combined with contextual vectors. A state is parameterized as:

$$s_t = \begin{bmatrix} \mathbf{P}_t \\ b_t \\ \vec{v}_t \\ \tau_t \end{bmatrix}$$

Where:

* **$\mathbf{P}_t$**: An $N \times 2$ matrix (where $N=36$ active players) tracking the global coordinate layout of every athlete on the field:

$$\mathbf{P}_t = \begin{bmatrix} x_{1,t} & y_{1,t} \\ x_{2,t} & y_{2,t} \\ \vdots & \vdots \\ x_{36,t} & y_{36,t} \end{bmatrix} \in \mathbb{R}^{36 \times 2}$$

* **$\mathbf{V}_t$**: Approximated as the discrete-time finite difference of the continuous position matrix function $\mathbf{P}(t)$. Given a standard tracking sampling interval of $\Delta t$, it acts as the empirical first derivative representing the instantaneous velocity all 36 players:

$$\mathbf{V}_t \approx \frac{d\mathbf{P}(t)}{dt} \approx \frac{\mathbf{P}_t - \mathbf{P}_{t-1}}{\Delta t} \in \mathbb{R}^{36 \times 2}$$

* **$b_t$**: the instantaneous postion of the ball.
* **$\vec{v}_t = (\Delta x, \Delta y)$**: represents the instantaneous velocity and directional vector of the ball.
* **$\tau_t$**: represents contextual game state metadata (e.g., time remaining in the quarter, point differential).
### 1.2 The Action Space ($E_t$)
We treat the action space as a discrete event set. It evolves from the continuous play of the game into a discrete event steam.

$E_t = \{(i, \text{event\_type})\}$
* i is the player id
* event_type is $\in$ (kick, handball, spoil, mark, tackle)

### 1.3 Transition Probabilities ($P$)
The transition dynamics of the environment are dictated by the probability distribution $P(s_{t+1} \mid s_t, e_t)$. This defines the probability that executing action $e_t$ from state $s_t$ will result in the system transitioning to state $s_{t+1}$. 
* In a chaotic, 360-degree environment like AFL, $P(s_{t+1} \mid s_t, e_t)$ captures both the physical execution error of the player and the chaotic intervention of opponents (e.g., spoils, effective tackling, wind conditions).

### 1.4 The Reward Function ($R$) and Horizon
Because AFL possessions are highly interdependent, rewards are modeled as **terminal and episodic**. Intermediate actions do not receive immediate extrinsic rewards ($R_t = 0$). The reward function resolves strictly when a possession chain terminates ($T$):

$R_{\text{terminal}} \in \{ +6.0, +1.0, 0.0, -1.0, -6.0 \}$

Where:
* $+6.0$: Chain ends in a Goal scored by the attacking team.
* $+1.0$: Chain ends in a Behind scored by the attacking team.
* $0.0$: Chain ends in a neutral stoppage (referee bounce/throw-in) or clean out-of-bounds.
* $-1.0$: Chain is turned over and results in an immediate opposition Behind.
* $-6.0$: Chain is turned over and results in an immediate opposition Goal.

Because every match is a fixed duration and every possession chain is finite, the discount factor is set to $\gamma = 1.0$. This ensures that the model optimizes for actual scoreboard outcomes without artificially diminishing the value of multi-stage setup play.

---

## 2. Value Estimation via Offline Reinforcement Learning

Because we cannot run active simulations to train a policy, we apply Offline Reinforcement Learning to a historical dataset. This is where the magic comes in for me. Using ORL we can capture the value of any given game configuration without any priors about what is good or bad.

### 2.1 The Multi-Agent State-Value Function $V(s_t)$

The State-Value Function $V(s_t)$ represents the expected terminal reward given the global configuration of the match at timestamp $t$. It maps the latent value of the entire field layout:

$$V(s_t) = \mathbb{E} \left[ R_{\text{terminal}} \mid S_t = s_t \right]$$

Because $s_t$ contains the player position matrix $\mathbf{P}_t$ and the velocity matrix $\mathbf{V}_t$, $V(s_t)$ does not just learn to evaluate the quality of the previous actions or where the ball is. It learns to evaluate the whole positional configuration of the game.


### 2.2 The Sparse Event Action-Value Function $Q(s_t, E_t)$

The Action-Value Function $Q(s_t, E_t)$ defines the expected terminal reward of entering state $s_t$ and observing the sparse event set $E_t$. It resolves under two distinct operational pathways based on our event-driven architecture:

$$Q(s_t, E_t) = \sum_{s_{t+1} \in S} P(s_{t+1} \mid s_t, E_t) V(s_{t+1})$$

#### Pathway A: Passive System Evolution ($E_t = \emptyset$)
When no technical event occurs, the $Q$-value measures the expected progression of the play based purely on physical momentum and tracking trajectories:

$$Q(s_t, \emptyset) = \mathbb{E}[V(s_{t+1}) \mid s_t]$$

#### Pathway B: Active Event Disruptions ($E_t = \{(i, \kappa, \mu)\}$)
When an explicit event $\kappa$ is executed by player $i$ (e.g., a kick or a spoil), the transition probability shifts non-linearly. The $Q$-value captures the expected value of the state immediately after the event's physical resolution:

$$Q(s_t, \text{Event}) = \sum_{s_{t+1}} P(s_{t+1} \mid s_t, i, \kappa, \mu) V(s_{t+1})$$

---

### 3. The Counterfactual Causal Layer

To isolate an individual player's execution and decision-making from structural team bias, I propose a counterfactual causal layer based on Structural Causal Models (SCMs). By conditioning our calculations on the multi-agent position and velocity matrices ($\mathbf{P}_t, \mathbf{V}_t$), the state space acts as a backdoor adjustment set, neutralizing the confounding effects of system quality.

### 3.1 Realized Action vs. Passive Counterfactual (Execution Value)

We evaluate the structural value of a physical intervention by comparing the observed active event $E_t$ against the hypothetical scenario where the player elected not to intervene ($E_t = \emptyset$, preserving passive physical tracking trajectory):

$$\alpha(s_t, E_t) = Q(s_t, E_t) - Q(s_t, \emptyset)$$

This metrics isolates pure physical execution capability. It answers: *Did the mechanical execution of this kick, handball, or tackle improve our field state relative to simply continuing to run or hold the ball?*

### 3.2 Observed Choice vs. Optimal Counterfactual (Decision Quality)

To evaluate a player's field vision and cognitive execution under chaotic pressure, we map the observed event against the maximum potential value of all counterfactual alternative valid events $E'_t$ contained in the local action envelope $\mathcal{E}(s_t)$:

$$\text{Decision Regret } (R_t) = Q(s_t, E_{\text{observed}}) - \max_{E' \in \mathcal{E}(s_t)} Q(s_t, E')$$

Where $\mathcal{E}(s_t)$ is calculated geometrically by casting ray-traces through the player position matrix $\mathbf{P}_t$ to identify open, un-interceptable passing lanes. 
* A player who consistently maintains a Decision Regret near $0.0$ is executing optimal spatial choices under pressure, regardless of whether their raw disposal count is high or low.

### 3.3 Observed Player Action vs Average Player Action 

I have some intuition that I haven't formalised here. You should be able to ask a question like "How important was player x to state transitions $G$?". Where $G$ is some state transitions leading to a favourable position or score. If you replaced the player with some average player and asked what would of happend, it should tell you how valuable player x was w.r.t $G$. I don't have a formalism for this yet but it anyone wants to help me out that would be awesome. 


## 4. Limitations
There are several key limitations to this approach that come from the underlying structure of invasion sport. Firstly, how do you get $P_t$ and $V_t$? It's easy to build a model on these concepts but actually having data for the coordinates of each player across a game is non-trivial. I imagine only AFL clubs have access to this kind of data. Meaning that my modelling idea is kind of stuck in theory land. 

Secondly, in casaul analysis you really want action distributions for each play given each state. This would create a massive sparsity issue. So in my model what ends up happening is the $P(s_{t+1}|...)$ actually just amalgamates the spatial decsion making of all players into one distribution. Which simplifies the maths but also the mark. Although, you can kind of kick the can down the road by saying, once we have the universal baseline of what action we would expect to see for a given $s_t$ you can measure how much a player deviates from that baseline leading to an increased $V(s_{t+1})$
