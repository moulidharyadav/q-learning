# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.

### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3:
After training, derive the optimal policy from the Q-table.

### Step 4:
Implement the Monte Carlo method to estimate state values.

### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name:moulidhar g
### Register Number: 212223240042
```
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="533" height="718" alt="Screenshot 2025-11-15 191252" src="https://github.com/user-attachments/assets/93d5c5ee-ec21-46ef-9d59-d42d1d04536d" />
<img width="558" height="621" alt="Screenshot 2025-11-15 191414" src="https://github.com/user-attachments/assets/c5e6fcc0-35ba-4836-8103-309d782d4c12" />
<img width="409" height="163" alt="Screenshot 2025-11-15 191521" src="https://github.com/user-attachments/assets/5fe09216-7233-4e51-80e0-3a699ead6585" />
<img width="556" height="612" alt="Screenshot 2025-11-15 191638" src="https://github.com/user-attachments/assets/8761f775-ec13-4af4-9b51-e66fcf1218ce" />
<img width="421" height="124" alt="Screenshot 2025-11-15 191740" src="https://github.com/user-attachments/assets/f55d7632-1e9a-4e85-9f66-3793d55666fb" />
<img width="1645" height="664" alt="image" src="https://github.com/user-attachments/assets/ea9f5c18-0db6-4d66-a2c2-d227e160d71c" />
<img width="1693" height="744" alt="image" src="https://github.com/user-attachments/assets/e1962d27-47c6-44ce-98dc-824730ddd3db" />


## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given RL environment using Q-Learning and compared the state values with the Monte Carlo method.
