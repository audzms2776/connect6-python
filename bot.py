import numpy as np
import random
from common import Point
import tensorflow as tf


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Bot:
    """ A bot class. Override this class to implement your own Connect6 AI. """

    def __init__(self, player=1):
        self.player = player

    def move(self, board, nth_move):
        raise NotImplementedError("Implement this to build your own AI.")
        pass

    @property
    def bot_kind(self):
        return self.__class__.__name__


class RandomBot(Bot):

    """ Example bot that runs randomly. """
    def move(self, board, nth_move):
        x = random.randrange(0, 19)
        y = random.randrange(0, 19)
        
        return x, y


class Player(Bot):
    """ 플레이어도 봇으로 취급하지만, 사용자로부터 입력을 받음. """

    def move(self, board, nth_move):
        move = input('turn : ')
        return Point.from_name(move)


class TFBot(Bot):

    def __init__(self, player):
        self.player = player
        self.X = tf.placeholder(tf.float32, [None, 19, 19, 1])
        conv_layer1 = tf.layers.conv2d(self.X, 32, kernel_size=(5, 5), 
            padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv_layer1, 
            pool_size=[2, 2], strides=2)
        flat_layer = tf.layers.flatten(pool1)

        dnn_layer1 = tf.layers.dense(flat_layer, 1024, activation=tf.nn.relu)
        dnn_layer2 = tf.layers.dense(dnn_layer1, 512, activation=tf.nn.relu)
        self.action_pred = tf.nn.softmax(tf.layers.dense(dnn_layer2, 19 * 19))

        self.Y = tf.placeholder(tf.float32, [None, 19 * 19], name="input_y")
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")

        log_lik = -self.Y * tf.log(self.action_pred)
        log_lik_adv = log_lik * self.advantages
        self.loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

        learning_rate = 0.0001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    

    def move(self, board, nth_move):
        state = np.array(board).reshape(1, 19, 19, 1)
        action_prob = self.sess.run(self.action_pred, feed_dict={self.X: state})

        nn = np.array(board).reshape(-1)
        mask_list = list(map(lambda x: int(x == 0), nn))
        mumu = mask_list * action_prob[0]
        mask_p = mumu / mumu.sum()

        action = np.random.choice(np.arange(19 * 19), p=mask_p)
        
        action_space = np.zeros(19 * 19)
        action_space[action] = 1
        result_space = action_space.reshape((19, 19))

        x, y = np.where(result_space == 1.0)

        # save
        self.xs = np.vstack([self.xs, state])
        self.ys = np.vstack([self.ys, action_space])

        return int(x), int(y)


    def reset_data(self):
        self.xs = np.empty(shape=[0, 19, 19, 1])
        self.ys = np.empty(shape=[0, 19 * 19])
        self.rewards = np.empty(shape=[0, 1])
        self.reward_sum = 0

    
    def train(self):
        dis_rewards = discount_rewards(self.rewards)
        dis_rewards = (dis_rewards - dis_rewards.mean()) / (dis_rewards.std() + 1e-7)

        # print(dis_rewards.shape)
        # print(self.xs.shape)
        # print(self.ys.shape)

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict= {
            self.X: self.xs,
            self.Y: self.ys,
            self.advantages: dis_rewards
        })

        return loss