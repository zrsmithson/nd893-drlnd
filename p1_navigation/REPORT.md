# Project 1
## Navigation

### Learning Algorithm
Q-learning is a reinforcement learning algorithm that does not require a reward function that can handle stochastic steps and rewards of a finite Markov decision process to find an optimal policy. A Deep Q-learner (DQN) uses a deep neural network with a nonlinear activation function to reach a pseudo-optimal policy, but with a large observation space, this leads to empirically better results. While DQNs can learn purely with pixel inputs through a convolutional neural network (CNN), the simplest implementation can use a simpler fully connected neural network to map environmental inputs to actions.

![Reinforcement Learning: Agent takes an action and observes the state and reward from the environment](images/rl_environment_action.png)

Every time the agent takes a step in the environment, it observes its new state and the reward it received. When it made the move, it had an expected reward/state, and the difference between expected and observed is known as the temporal difference (TD) error. In Q-learning, an associative learning update rule factors in the next expected reward (based on best expected action), discounted by a factor `γ`.

![TD error](https://render.githubusercontent.com/render/math?math=TDerror%20%3D%20target-expected%20%3D%20r_%7Bt%2B1%7D%2B%5Cgamma%20%5Cmax%5Climits_%7Ba'%7DQ(s_%7Bt%2B1%7D%2Ca')-Q(s_t%2Ca_t))


The Q values are updated based on this error.

![Q update function](https://render.githubusercontent.com/render/math?math=Q(s_t%2Ca_t)%20%5Cleftarrow%20Q(s_t%2Ca_t)%20%2B%20%5Calpha%20%5Cleft%20%5B%20r_%7Bt%2B1%7D%2B%5Cgamma%20%5Cmax%5Climits_%7Ba'%7DQ(s_%7Bt%2B1%7D%2Ca')-Q(s_t%2Ca_t)%5Cright%20%5D)



A vanilla DQN uses 4 techniques to overcome the theoretical instability issue, with the primary benefit coming from Experience Replay and Fixed Target Network.

#### Experience Replay
This samples a small batch of states and actions into a buffer, which helps to decouple the dependant variables and avoid overfitting. This also increases the learning speed by "remembering" past rare events to avoid only learning from the most common observations, making better use of experiences. In short, errors due to consecutive experiences is mitigated by sampling experiences randomly, out of order.

#### Fixed Target Network
To avoid the network changing too rapidly, two networks of the same architecture are used to "lead" the local network more slowly. This can be done by updating the target network's weight less often (or more slowly) than the local network.

![Q local update function](https://render.githubusercontent.com/render/math?math=Q_%7Blocal%7D(s_t%2Ca_t)%20%5Cleftarrow%20%5Calpha%20%5Cleft%20%5B%20r_%7Bt%2B1%7D%2B%5Cgamma%20%5Cmax%5Climits_%7Ba'%7DQ_%7Btarget%7D(s_%7Bt%2B1%7D%2Ca')%20%5Cright%20%5D%20%2B%20(1-%5Calpha)%20%5Cleft%20%5B%20Q_%7Blocal%7D(s_t%2Ca_t)%5Cright%20%5D)

![Q target update function](https://render.githubusercontent.com/render/math?math=Q_%7Btarget%7D(s_t%2Ca_t)%20%5Cleftarrow%20%5Ctau%20%5Cleft%20%5BQ_%7Blocal%7D(s_t%2Ca_t)%20%5Cright%20%5D%20%2B%20(1-%5Ctau)%20%5Cleft%20%5B%20Q_%7Btarget%7D(s_t%2Ca_t)%5Cright%20%5D)

#### Clipping Rewards
Often when positive rewards have a varying value, training stability can be increased by simplifying positive and negative rewards to +1 and -1. respectively.

#### Skipping Frames
An agent acting in an environment designed for humans does not need to make 60 decisions per second. Frames can be skipped for acting, learning, or both.

### Methods

The implemented agent uses Experience Replay with the buffer and batch size specified as hyperparameters. A fixed target network was used, where a slowly updating target network was used with tau specified as a hyperparameter. The reward did not need to be clipped as the environment provided rewards of -1 and 1 by default. Frames were skipped on the learning side to reduce computational complexity, but the action decisions were made every frame.

#### Hyperparameters
| Name          | Value   | Description                               |
|---------------|---------|-------------------------------------------|
| BUFFER_SIZE   | 100,000 | Replay buffer size                        |
| BATCH_SIZE    | 64      | Minibatch size                            |
| GAMMA         | 0.99    | Discount factor                           |
| TAU           | 0.001   | Soft update of target parameters          |
| LR            | 0.0005  | Learning rate                             |
| UPDATE_EVERY  | 4       | How often to update the network (steps)   |
| n_episodes    | 2000    | Max number of episodes in learning loop   |
| max_t         | 1000    | Max number of steps in an episode         |
| eps_start     | 1.0     | Start value of epsilon                    |
| eps_end       | 0.01    | End value of epsilon                      |
| eps_decay     | 0.995   | Decay of epsilon until epsilon=eps_end    |

#### Network Architecture
![Network Architecture](images/network_architecture.png)


### Results
Training took 563 episodes to reach an average score of 13, but I trained the model to 15 since it did not seem to be overfitting.

```
Episode 100	Average Score: 1.05
Episode 200	Average Score: 4.03
Episode 300	Average Score: 7.60
Episode 400	Average Score: 10.37
Episode 500	Average Score: 12.84
Episode 600	Average Score: 14.17
Episode 663	Average Score: 15.01
Environment solved in 563 episodes!	Average Score: 15.01
```

![Trained Agent](images/zs_banana.gif)

![score vs episode number](images/score_episode.png)

### Improvements
From watching the results, the agent would occasionally oscillate back and forth, and stop going toward a new banana. By expanding the Skipping Frames technique to also skip frames when exercising the agent, it might remove some of the unnecessarily quick decision-making. I also think it would be beneficial to explore a [Double DQN](https://arxiv.org/abs/1509.06461), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), or [Dueling DQN](https://arxiv.org/abs/1511.06581) to see if it would improve the training performance. It might be beneficial to then implement an algorithm similar to [Rainbow](https://arxiv.org/abs/1710.02298) where the best of each model can be used at appropriate times to improve performance and training time.

Beyond this, it would be worth exploring the visual input into the system instead of just looking at the dimensional data provided from the nearby bananas. Using a CNN to map pixels to actions might lead to better results, but it would likely be a more realistic exercise for something in robotics or game-playing, since this is what available to a human operator. In this way, the skill could be compared to human level to know when the agent achieves superhuman banana collecting.