#Tensorflow 2.0
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import tensorflow_probability as tfp
import tf_agents as tfa

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common



def build_actor_net(observation_space,action_space):
    out_dim=1 if action_space.shape ==() else action_space.shape
    fc_layers2 = [7*7*64,512]
    actor_net = actor_distribution_network.ActorDistributionNetwork(observation_space,action_space,fc_layer_params = fc_layers2)
    return actor_net

pass
def build_value_net(observation_space):
    fc_layesr=[7*7*64,512]
    q_net =ValueNetwork(observation_space
                        ,fc_layer_params=fc_layesr,
                    dropout_layer_params=[1,3])

    return q_net
#   To measure Performance
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]






















if __name__ == '__main__':
    # En teoría deberia estar haceindo todo lo que pueda en la GPU no se si lo esta haciendo
    with tf.device("GPU:0"):

        env_name = 'CartPole-v0'


        ATEMP = 0
        NAME = "ATEMP_%i" % ATEMP
        score_prom = None
        writer = tf.summary.create_file_writer("logs\\"+NAME)
        env = suite_gym.load(env_name)

        train_env = tf_py_environment.TFPyEnvironment(env)


        agent_net=build_actor_net(train_env.observation_spec(),train_env.action_spec())
        value_net= build_value_net(train_env.observation_spec())

        train_step_counter = tf.compat.v2.Variable(0)


        #llamo al agetne que realiza Proximal policy optimization
        agent= tfa.agents.ppo.ppo_agent.PPOAgent(time_step_spec=train_env.time_step_spec(),
                                                 action_spec= train_env.action_spec(),
                                                 optimizer= tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
                                                 ,actor_net= agent_net
                                                 ,value_net=value_net
                                                 ,train_step_counter=train_step_counter)
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,batch_size=1)
        #Wrapping the function so it act like a graph
        agent.train = common.function(agent.train)



        driver =dynamic_episode_driver.DynamicEpisodeDriver(
            train_env,agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_episodes=1,
        )


        #Loop for training the  agent


        for step in range(100):


            driver.run()
            experience= replay_buffer.gather_all()
            agent.train(experience)
            replay_buffer.clear()
            print(f"Episode: {step:d}")



            if step%10==0:
                with writer.as_default():
                    performance = compute_avg_return(train_env,agent.policy,num_episodes=10)
                    tf.summary.scalar("Performance",performance,step=step)
                    writer.flush()
                    print(f"Performance: {performance:0.3f}")




        writer.close()



















