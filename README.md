# Frozen Lake 구현하기

수업시간에 배운 Frozen Lake를 구현해보았습니다.  
팀원 둘 다 Python을 거의 다루지 못해 매우 부족하지만.. 열심히 작성해보았습니다..
아나콘다 jupyter notebook에서 작성하였습니다.
<br>
<br>
<br>


> 먼저 Frozen Lake V0을 불러옵니다.
```import numpy as np
import gym
import random
import time
from IPython.display import clear_output
env = gym.make("FrozenLake-v0")
```

> action space, 그리고 state space, Q table에 대해 정의하고
Hyperparameter를 설정합니다.

```action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
 
num_episodes = 40000
max_steps_per_episode = 1000
learning_rate = 0.1
discount_rate = 0.99

rewards_all_episodes = []
```


> action 전략 ① Epsilon Greedy 접근법 정의
```def epsilonGreedyExplore(env, state, Q_table, e, episodes):
    prob = 1 - e / episodes
    if np.random.rand() < prob:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
    return action
```
		
> action 전략 ② 볼츠만 접근법(softmax explore) 정의
```def softmaxExplore(env, state, Q_table, tau=1):
    num_action = env.action_space.n
    action_prob = np.zeros(num_action)
    denominator = np.sum(np.exp(Q_table[state, :] / tau))

    for a in range(num_action):
        action_prob[a] = np.exp(Q_table[state, a] / tau) / denominator
    action = np.random.choice([0, 1, 2, 3], 1, p=action_prob)[0]
    return action
```
<br>

> 강화학습 방법론 중 Q-Learning 알고리즘을 적용합니다.<br>
이 때, action 전략 중 epsilon-greedy를 먼저 실행합니다.

```
# Q-learning 학습

# Epsilon-Greedy 탐색 시
strategy = "epsilon-greedy"

for episode in range(num_episodes):
    # 새로운 에피소드 초기화
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):

        if strategy == "epsilon-greedy":
            action = epsilonGreedyExplore(env, state, q_table, episode, num_episodes)
        else:
            action = softmaxExplore(env, state, q_table)

        new_state, reward, done, info = env.step(action)
        
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break
    rewards_all_episodes.append(rewards_current_episode)
```

   
> episode를 총 4만번 돌리므로, 2,000번의 에피소드당 평균 성공 확률을 표기합니다.
그리고 update된 q table을 출력합니다.

```
# 2000번의 에피소드당 평균 성공 확률
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/2000)
count = 2000
 
print("********2000 에피소드당 평균 reward ********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/2000)))
    count += 2000
 
# 업데이트된 q_table을 출력
print("\n\n********Q_table********\n")
print(q_table)
```

> 학습이 완료되었다면, episode를 돌려 봅니다.
무사히 Goal에 도달한 것을 확인할 수 있습니다.
```
for episode in range(3):
    # 각 에피소드의 변수를 초기화한다.
    state = env.reset()
    done = False
    print("*****에피소드 ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        # 현재 상태를 그려 본다.
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        # 현재 상태에서의 q값(보상)이 가장 큰 action을 취한다.
        action = np.argmax(q_table[state, :]) 
        # 새로운 action을 취한다.
        new_state, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                # 만약에 Goal에 도착하여 reward가 1이라면
                print("****목표에 도달하였습니다.!****")
                time.sleep(3)
            else:
                # Goal에 도달하지 못했다면
                print("****Hole에 빠지고 말았습니다.****")
                time.sleep(3)
                clear_output(wait=True)            
            break
        
        # 새로운 상태를 설정한다.
        state = new_state
env.close()
```

![](https://github.com/youngseo-0203/RL/blob/main/%EB%AA%A9%ED%91%9C%EB%8F%84%EB%8B%AC.png)

<br/>

> 다음으로 볼츠만 함수 적용을 위해 strategy를 softmax로 바꾸어줍니다.

```
# 볼츠만(소프트맥스) 탐색 시
strategy = "softmax"

# Q-learning 학습
for episode in range(num_episodes):
    # 새로운 에피소드 초기화
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):

        if strategy == "epsilon-greedy":
            action = epsilonGreedyExplore(env, state, q_table, episode, num_episodes)
        else:
            action = softmaxExplore(env, state, q_table)

        new_state, reward, done, info = env.step(action)
        
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break
    rewards_all_episodes.append(rewards_current_episode)
```


> 동일하게 2,000번의 에피소드당 성공 확률과 업데이트된 Q table을 출력해봅니다.

```
# 2000번의 에피소드당 평균 성공 확률
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/2000)
count = 2000
 
print("********2000 에피소드당 평균 reward ********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/2000)))
    count += 2000
 
# 업데이트된 q_table을 출력
print("\n\n********Q_table********\n")
print(q_table)
```


> 학습이 완료되었으면, 에피소드를 실행해봅니다. 무사히 Goal에 도달한 것을 확인할 수 있습니다

```
for episode in range(3):
    # 각 에피소드의 변수를 초기화한다
    state = env.reset()
    done = False
    print("*****에피소드 ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        # 현재 상태를 그려 본다.
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        # 현재 상태에서의 q값(보상)이 가장 큰 action을 취한다.
        action = np.argmax(q_table[state, :]) 
        # 새로운 action을 취한다
        new_state, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                # 만약에 Goal에 도착하여 reward가 1이라면
                print("****목표에 도달하였습니다.!****")
                time.sleep(3)
            else:
                # Goal에 도달하지 못했다면
                print("****Hole에 빠지고 말았습니다.****")
                time.sleep(3)
                clear_output(wait=True)            
            break
        
        # 새로운 상태를 설정한다.
        state = new_state
env.close()
```

![](https://github.com/youngseo-0203/RL/blob/main/%EB%AA%A9%ED%91%9C%EB%8F%84%EB%8B%AC.png)
