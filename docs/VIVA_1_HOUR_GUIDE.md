# EdgeBot — 1-Hour Viva Guide

Read this top to bottom once (about 40 min), then spend 20 min saying the answers out loud.
Short report: [`project_report.pdf`](project_report.pdf) (12 pages).

---

## ⭐ Memory cheat sheet (learn this first)

### Start the app (one command)
```bash
make web
```
Then open **http://localhost:8000** in Chrome.

### 5 commands that always work (after **Reset warehouse**, in this order)
1. `hello jetbot`
2. `move the red box from shelf a to the conveyor`
3. `move the yellow parcel from shelf b to the packing station`
4. `what is your status`
5. `go back to the charging dock`

Bonus: `pick up the box` → robot asks "Which colour?" → answer `the blue one`.

### Algorithms: remember **"Q → D → P → S"** (each fixes the one before)
| Letter | Algorithm | One line | Its problem → next |
|---|---|---|---|
| **Q** | Q-learning (and SARSA) | Table of Q(s,a), update with the TD error | Table can't hold continuous lidar → **D** |
| **D** | DQN | Neural net replaces the table: replay buffer + target net + Double DQN | Only 5 fixed moves → **P** |
| **P** | **PPO** (used on the robot) | Actor-critic, clipped ratio ±20%, continuous speed + turn | Still crashes sometimes → **S** |
| **S** | **Shield** (my novelty) | Predict 0.5 s ahead with lidar; block unsafe moves | Collisions → **0%** |

### Papers: remember **"2 classics + 2 safety + 2 compression + 2 recent"**
| Group | Paper (year) | Why I used it |
|---|---|---|
| Classic | **DQN**, Mnih et al., *Nature* (2015) | Value-based deep RL |
| Classic | **PPO**, Schulman et al. (2017) | My deployed algorithm |
| Safety | **Shielding**, Alshiekh et al., AAAI (2018) | Original shield idea |
| Safety | **Realizable Continuous Shields** (2024) | Shields for PPO + lidar, but they need a model |
| Compression | **QuaRL**, Krishnan et al. (2022) | Quantised RL for edge chips |
| Compression | **Quantized Continuous Controllers** (2025) | 8-bit RL controllers |
| Recent | **REASAN** (2025) | Learned safety shield with one lidar |
| Recent | **Lightweight Safe RL for UAV** (2026) | Safe RL with sparse lidar |

**My gap in one sentence:** "Their safety filters need a map, a model or training. Mine needs only the lidar and the robot's equations, and I tested it on 8-bit compressed policies."

---

## Study plan (60 min)

| Time | What to learn | Section |
|---|---|---|
| 0–5 min | The 30-second pitch | §1 |
| 5–15 min | Why software only, the MDP | §2, §3 |
| 15–35 min | The 3 algorithms: Q-learning → DQN → PPO | §4 |
| 35–45 min | Your novelty (the shield) and its numbers | §5 |
| 45–50 min | Papers you used | §6 |
| 50–60 min | Practice the hard questions out loud | §7 |

---

## 1. The 30-second pitch (memorise)

> "EdgeBot is a voice-commanded warehouse robot, built fully in simulation. You say *move the red box
> from shelf A to the conveyor*. Whisper turns speech to text, a small BERT model extracts the intent and
> slots, A\* plans a route, and a **PPO reinforcement-learning policy** drives the robot using a 24-beam
> lidar while avoiding clutter. My novelty is a **lidar safety shield** between the policy and the
> motors. It needs no map and no retraining. It brought collisions to **0%** for both normal (FP32) and
> compressed edge (INT8) policies."

**The four questions the professor will ask:**
1. **What is the project?** The pitch above.
2. **What problem does it solve?** A robot driving through a warehouse whose layout keeps changing. You
   cannot hand-code every case, so it learns a driving policy. Learned policies, especially once
   compressed to INT8 for small edge chips, can still crash sometimes. The shield stops that.
3. **Which RL algorithm?** **PPO** is deployed. DQN is the baseline. Q-learning/SARSA are the
   foundation.
4. **Which papers?** PPO (Schulman 2017), DQN (Mnih 2015), Double DQN (van Hasselt 2016), shielding
   (Alshiekh 2018), plus 2024–2026 work on continuous shields, quantised RL and safe lidar navigation
   (§6).

---

## 2. Why software only (no hardware)

Say it confidently. This was a design decision, not a shortcut.

- **RL learns by trial and crash.** PPO needed **5 million steps** and thousands of collisions. On a
  real robot that means weeks of driving and broken parts. In simulation it took 25 minutes.
- **Fair comparison.** Every algorithm is tested on the **same 300 unseen warehouses** with fixed random
  seeds. You cannot reset a real floor 300 times.
- **The course is about algorithms.** MDP, TD, DQN and PPO are all software.
- **It is still realistic.** It uses the standard Gymnasium API, a real JetBot footprint (12 cm radius),
  a 24-beam lidar and unicycle physics. The models are exported to ONNX/INT8, the format a Jetson runs.
- **Honest limitation:** latency was measured on a laptop CPU, not on a real Jetson. Sim-to-real is
  future work.

---

## 3. The MDP (write this on the board)

| | Meaning in EdgeBot |
|---|---|
| **State s** (29 numbers) | 24 lidar distances + distance to waypoint + sin/cos of bearing + previous action (2) |
| **Action a** | speed v ∈ [0, 0.5] m/s, turn rate ω ∈ [−2, 2] rad/s |
| **Transition P** | unicycle: x += v·cosθ·dt, y += v·sinθ·dt, θ += ω·dt (dt = 0.1 s) |
| **Reward R** | +3 × progress − 0.01 per step − small penalty near obstacles − jerk penalty; **+10 goal, −10 crash** |
| **γ** | 0.99 |

**Why is it Markov?** The lidar, the goal direction and the last action together describe everything
needed to choose the next action. The past adds nothing more.

**Why reward progress (shaping) and not only +10 at the goal?** With only a sparse reward, the robot
almost never reaches the goal by chance, so it gets nothing to learn from. The progress reward gives a
learning signal on every step.

---

## 4. The three algorithms (each one fixes the previous one's limit)

### 4.1 Q-learning: tabular TD control (foundation)
Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
- The term in brackets is the **TD error**: (what I got + what I expect next) − what I predicted.
- **Off-policy**: it learns the greedy policy while it explores with ε-greedy.
- **SARSA** uses Q(s',a') of the action *actually taken* instead of the max, so it is **on-policy**.
- **Result on the grid warehouse with an oil spill:** Q-learning learns the shortest edge route (return
  −11, optimal), and SARSA learns a safer route further from the spill. This is the cliff-walking example
  from Sutton & Barto. Value iteration (DP) gives the exact answer, and the two agree.
- **Limit:** a table cannot store a continuous 29-D lidar state, so we approximate Q with a neural net.

### 4.2 DQN: deep Q-network (value-based deep RL)
Loss = ( r + γ·Q_target(s', argmax_a' Q_online(s',a')) − Q_online(s,a) )²
- **Replay buffer:** train on random old transitions. This breaks the correlation between consecutive
  steps.
- **Target network:** a slowly updated copy that gives the network a stable target.
- **Double DQN:** the online net *chooses* a', the target net *evaluates* it. This reduces
  over-estimation from the max.
- **Dueling:** Q = V(s) + A(s,a), which learns how good a state is separately from how good each action
  is.
- Uses 5 discrete moves (forward, forward-left, forward-right, spin left, spin right). The project also
  has a CNN version on a 64×64 image ("RL for vision").
- **Result:** 86% success (lidar), 84% (pixels).
- **Limit:** only discrete actions, while real motors take continuous speeds.

### 4.3 PPO: proximal policy optimisation (DEPLOYED)
- **Actor-critic.** The actor outputs a Gaussian over (v, ω). The critic estimates V(s).
- **Advantage Â** = how much better an action was than average, computed with **GAE** (λ trades bias
  against variance).
- **Ratio** ρ = π_new(a|s) / π_old(a|s).
- **Objective:** maximise min( ρ·Â , clip(ρ, 0.8, 1.2)·Â ).
- **What the clip means:** if an update tries to change the probability of an action by more than ±20%,
  the gradient stops. This makes PPO stable without the complex maths of TRPO.
- Trained with 16 parallel environments for 5M steps (about 25 min).
- **Result:** **96% success**, 2% collision on 300 unseen layouts. The INT8 actor is 26 KB.

**Why PPO over DQN?** It handles continuous actions, gives smoother driving, has higher success (96% vs
86%), and is stable to train.

**Hierarchy:** A\* gives waypoints (the global route) and PPO follows them while dodging clutter. Rule:
*learn what is hard to model, compute what is known exactly.*

---

## 5. YOUR NOVELTY: Lidar safety shield for quantised edge policies

**Gap in the literature:**
- (a) Compressing RL policies to INT8 for edge chips can change their actions and hurt them (QuaRL 2022,
  Quantization & Pruning on DRL 2024, Quantized continuous controllers 2025).
- (b) Shields make RL safer (Alshiekh 2018, Realizable continuous shields 2024, REASAN 2025, UAV safe RL
  2026). But they usually need a known model or map, or are another learned network, and are tested on
  full-precision policies.
- **My idea:** a **map-free, training-free, kinematic lidar shield** used as a **safety net for INT8
  edge policies**, tested FP32 vs INT8 side by side inside a voice-controlled robot.

**How it works (`edgebot/rl/shield.py`, about 60 lines). Explain it in 4 steps:**
1. The policy (FP32 or INT8) proposes an action (v, ω).
2. The shield converts the current lidar scan into obstacle points around the robot.
3. It **simulates the next 5 steps (0.5 s)** with the unicycle equations, holding that action.
4. If any predicted position comes within **robot radius + 4 cm** of an obstacle, the action is unsafe.
   The shield replaces it with the *closest safe* candidate (forward, turn, slow or spin). A pure spin
   never moves the robot, so a safe option always exists.

**Results (300 unseen layouts, `results/shield_eval.json`):**

| Policy | Collision without → with shield | Success without → with shield | Shield active |
|---|---|---|---|
| PPO FP32 | 2.0% → **0%** | 96.0% → 96.0% | 5.5% of steps |
| PPO INT8 | 3.0% → **0%** | 95.7% → **97.0%** | 4.6% |
| DQN pixels FP32 | 1.0% → **0%** | 83.7% → 83.3% | 7.2% |
| DQN pixels INT8 (dynamic) | 1.3% → **0%** | 47.0% → 46.3% | 3.6% |
| DQN pixels INT8 (static) | 1.0% → **0%** | 73.7% → 72.3% | 5.6% |

**How to read it:**
- Collisions drop to zero for every model, and the shield acts only about 5% of the time.
- INT8 quantisation made PPO crash more (2% → 3%), and the shield removed all of those crashes.
- **Honest limit:** the shield stops *crashes*, not *getting lost*. The badly quantised DQN fails mostly
  by **timeouts** (dithering). A shield cannot fix that; better quantisation (static INT8) does. A few
  crash episodes become timeouts, and stopping safely is the right way for a real robot to fail.

**Why not train a safer policy instead (Safe RL / CMDP / Lagrangian)?** That needs retraining and still
gives no guarantee at every step. The shield is a hard per-step check, works on *any* policy, and costs
almost nothing on the edge device.

---

## 6. Papers (know one line about each)

| Paper | Year | What I took |
|---|---|---|
| Sutton & Barto, *RL: An Introduction* | 2018 | MDP, DP, Q-learning, SARSA |
| Mnih et al., DQN, *Nature* | 2015 | replay buffer + target network |
| van Hasselt et al., Double DQN, AAAI | 2016 | fixes over-estimation |
| Schulman et al., PPO, arXiv:1707.06347 | 2017 | clipped objective (deployed) |
| Schulman et al., GAE, ICLR | 2016 | advantage estimation |
| Alshiekh et al., Safe RL via Shielding, AAAI | 2018 | the original idea of a shield |
| Krishnan et al., QuaRL, TMLR | 2022 | quantising RL policies for edge |
| Impact of Quantization & Pruning on DRL, arXiv:2407.04803 | 2024 | compression can hurt RL policies |
| Realizable Continuous-Space Shields for Safe RL, arXiv:2410.02038 | 2024 | shields for continuous actions; PPO + lidar navigation |
| Learning Quantized Continuous Controllers for Integer Hardware, arXiv:2511.07046 | 2025 | INT8 RL controllers on embedded chips |
| REASAN: Reactive Safe Navigation for Legged Robots, arXiv:2512.09537 | 2025 | RL safety shield with a single lidar |
| Embodied Foundation Models at the Edge (survey), arXiv:2603.16952 | 2026 | edge deployment constraints |
| Lightweight Safe RL for End-to-End UAV Navigation, arXiv:2607.01794 | 2026 | safe RL under sparse lidar |

**Difference from REASAN / UAV 2026:** their safety module is *learned* (another network, which needs
training). Mine is *analytic*: kinematics plus lidar, with zero training. It is also tested specifically
on INT8 policies.

**Difference from Realizable Shields 2024:** theirs uses a formal model of the environment. Mine uses
only the current lidar scan.

---

## 7. Hard questions and answers

- **Is the shield really novel?** Shielding itself is not new (2018). The new part is the combination:
  map-free, training-free, used to protect INT8-quantised edge policies, and measured FP32 vs INT8.
  Say *"to the best of my knowledge"*. Never claim "nobody ever did this".
- **On-policy vs off-policy?** PPO and SARSA are on-policy: they learn from the current policy's own
  data. Q-learning and DQN are off-policy: they can reuse old data from a replay buffer.
- **Why does PPO use a critic?** To compute the advantage. Subtracting a baseline V(s) lowers the
  variance of the policy gradient.
- **Exploration?** DQN uses ε-greedy (ε decays). PPO samples from its Gaussian; the learned standard deviation shrinks as it gets confident (entropy bonus is 0 in my config).
- **What is GAE λ?** λ = 0 uses the one-step TD error (low variance, more bias). λ = 1 uses the Monte
  Carlo return (no bias, high variance). I used 0.95, with γ = 0.99 and clip ε = 0.2.
- **Why did dynamic INT8 break the pixel DQN?** Per-tensor dynamic scaling crushes the small CNN
  activations, which changes which Q-value is the max. Static calibrated per-channel INT8 recovers most
  of it (47% → 74%).
- **Can the shield fail?** Yes. A 24-beam lidar can miss a thin pole between beams, it assumes static
  obstacles, and the 0.5 s horizon is short. Future work: a moving-obstacle prediction and a denser
  lidar.
- **Sim-to-real?** Use domain randomisation (sensor noise, friction), then run on a JetBot/Jetson with
  TensorRT. The ONNX files are ready.
- **How did you evaluate fairly?** Evaluation seeds start at 1,000,000 so they never overlap the
  training layouts. All 300 episodes are identical across methods.
- **What is Bellman optimality?** Q\*(s,a) = E[ r + γ·max_a' Q\*(s',a') ]. Q-learning is a sampled,
  incremental version of this equation.

## 8. Show it working
Run `make web` and open http://localhost:8000. Press **Reset warehouse**, click commands 1 → 5, then press
**Run on 30 unseen warehouses** to show collisions dropping to 0% with the shield on. Untick the shield box
to show the same robot running without the novelty.
