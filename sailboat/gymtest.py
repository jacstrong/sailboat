import gym
from gym import envs
from gym.envs.registration import register

import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import time

register(
    id='sailboat-v0',
    entry_point='envs:Sailboat',
)

np.random.seed(123)
env = gym.make('sailboat-v0')
# env = gym.make('CartPole-v0')
nb_actions = env.action_space.shape[0]

print('action space', env.action_space.shape[0])
# print(env.observation_space)

# model = Sequential()
# # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Flatten(input_shape=(1,14)))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,14)))
actor.add(Dense(30))
actor.add(Activation('relu'))
actor.add(Dense(20))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('softsign'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=2000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.6, mu=0, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=10000,
                  random_process=random_process, gamma=.1, target_model_update=1e-3)
# agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                   memory=memory, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=10000,
#                   random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=0.001,  clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something!
start = time.monotonic()
for x in range(1):
    agent.fit(env, nb_steps=200, visualize=False, verbose=0, nb_max_episode_steps=150, )
    print(x)
end = time.monotonic()
print(end-start)
agent.test(env, nb_episodes=2, visualize=True, nb_max_episode_steps=1000)

mode = 'train'
if mode == 'train':
    filename = 'test'
    # we save the history of learning, it can further be used to plot reward evolution
    # with open('_experiments/history_ddpg__redetorcs'+filename+'.pickle', 'wb') as handle:
        #  pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #After training is done, we save the final weights.
    # agent.save_weights('h5f_files/ddpg_{}_weights.h5f'.format('test'), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
elif mode == 'test':
    # env.set_test_performace() # Define the initialization as performance test
    # env.set_save_experice()   # Save the test to plot the results after
    agent.load_weights('h5f_files/ddpg_{}_weights.h5f'.format('test'))
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=1000)

# env.reset()
# # env.render()
# done = False
# print(done)
# count = 0
# while done == False:
#     if count < 15:
#         observation, reward, done, info = env.step([0, 0, 0, 0])
#     elif count < 30:
#         observation, reward, done, info = env.step([0, 0.5, 0, 0])
#     elif count < 60:
#         observation, reward, done, info = env.step([0, 0, 0, 0])
#     else:
#         observation, reward, done, info = env.step([0, 0.5, 0, 0])
#     # print(reward)
#     count += 1
#     pass
# env.render()
# print("count: " + str(count))

