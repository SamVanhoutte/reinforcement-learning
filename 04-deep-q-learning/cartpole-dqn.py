import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import collections

# Import Tensorflow libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

###### Tensorflow-GPU ########

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DQAgent:

    def __init__(self, replayCapacity, inputShape):
        ## Initialize replay memory
        self.capacity = replayCapacity
        self.memory = collections.deque(maxlen=self.capacity)
        self.populated = False
        ## q network
        self.inputShape = inputShape
        self.q_model = self.buildNetwork()

        ## Target network
        self.target_model = self.buildNetwork()
        self.target_model.set_weights(self.q_model.get_weights())

    def addToReplayMemory(self, step):
        self.step = step
        self.memory.append(self.step)

    def sampleFromReplayMemory(self, batchSize):
        self.batchSize = batchSize
        if self.batchSize > len(self.memory):
            self.populated = False
            return self.populated
        else:
            return random.sample(self.memory, self.batchSize)


    def buildNetwork(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.inputShape, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(24, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr = 0.001), metrics=['MeanSquaredError'])
        return model

    def q_network_fit(self,batch, batchSize):
        self.batchSize = batchSize
        self.batch = batch


    def q_network_predict(self, state):
        self.state = state
        self.qPolicy = self.q_model.predict(self.state)
        return self.qPolicy

    def target_network_predict(self, state):
        self.state = state
        self.qTarget = self.target_model.predict(self.state)
        return self.qTarget

    def update_target_network(self):
        self.target_model.set_weights(self.q_model.get_weights())



# Model parameters

DISCOUNT = 0.90
REPLAY_MEMORY_CAPACITY = 10000
#MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
BATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_INTERVAL = 500
EPSILON = 0.95 # Exploration percentage
MIN_EPSILON = 0.01
POSSIBLE_ACTIONS = [0,1]
DECAY = 0.999



# Create Cartpole environment
env = gym.make('CartPole-v0')
state = env.reset()
#state = env.reset()
done =  False

# create DQN agent

agent = DQAgent(replayCapacity= REPLAY_MEMORY_CAPACITY, inputShape= state.shape)


# Fill the replay memory with the first batch of samples

updateCounter = 0
rewardHistory = []


for episode in range(200):
    episodeReward = 0
    stepCounter = 0  # count the number of successful steps within the episode

    #print('\n', episode)
    state = env.reset()
    done = False
    #state = np.expand_dims(state, axis=0)

    while not done :
        env.render()

        r = random.random()

        if r <= EPSILON:
            action = random.sample(POSSIBLE_ACTIONS, 1)[0]
            #print('exploration')
        else:
            #print('exploitation')
            qValues = agent.q_network_predict(state.reshape(1,-1))
            action = np.argmax(qValues[0] )
            #print('action =', action)
            #print(qValues)

        newState, reward, done, info = env.step(action)

        if (done) and (stepCounter <199):
            reward = -10

        stepCounter +=1
        #print('stepcounter = ', stepCounter)

        #newState = np.expand_dims(newState, axis=0)
        # store step in replay memory
        step = (state, action, reward, newState, done)
        agent.addToReplayMemory(step)
        state = newState
        episodeReward += reward
        #print('episodeReward = ',episodeReward)
        # When enough steps in replay memory -> train policy network
        if len(agent.memory) >= (BATCH_SIZE ):
            EPSILON = DECAY * EPSILON
            if EPSILON < MIN_EPSILON:
                EPSILON = MIN_EPSILON
            # sample minibatch from replay memory
            miniBatch = agent.sampleFromReplayMemory(BATCH_SIZE)
            miniBatch_states = np.asarray(list(zip(*miniBatch))[0],dtype=float)
            miniBatch_actions = np.asarray(list(zip(*miniBatch))[1], dtype = int)
            miniBatch_rewards = np.asarray(list(zip(*miniBatch))[2], dtype = float)
            miniBatch_next_state = np.asarray(list(zip(*miniBatch))[3],dtype=float)
            miniBatch_done = np.asarray(list(zip(*miniBatch))[4],dtype=bool)

            current_state_q_values = agent.q_network_predict(miniBatch_states)
            y = current_state_q_values
            #print(y.shape)
            #miniBatch_next_state = np.squeeze(miniBatch_next_state, axis =1)

            next_state_q_values = agent.target_network_predict(miniBatch_next_state)
            #print(next_state_q_values.shape)
            max_q_next_state = np.max(next_state_q_values,axis=1)

            for i in range(BATCH_SIZE):
                if miniBatch_done[i]:
                    y[i,miniBatch_actions[i]] = miniBatch_rewards[i]
                else:
                    y[i,miniBatch_actions[i]] = miniBatch_rewards[i] + DISCOUNT * max_q_next_state[i]

            agent.q_model.fit(miniBatch_states, y, batch_size=BATCH_SIZE, verbose = 0)
            #print(y)

        else:
            env.render()
            continue
        if updateCounter == UPDATE_TARGET_INTERVAL:
            agent.update_target_network()
            print('target updated')
            updateCounter = 0
        updateCounter += 1
    print('episodeReward for episode ', episode, '= ', episodeReward, 'with epsilon = ', EPSILON)
    rewardHistory.append(episodeReward)

env.close()

plt.plot(rewardHistory)
plt.show()


#actions = agent.policy_network_predict(state)

#action = np.argmax(actions)
#print(action)

#state, reward, done, info = env.step(action)
#print(reward)


