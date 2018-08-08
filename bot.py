import numpy as np
import random
from common import Point

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
        move = input('{} turn : '.format(STONE_NAME[self.player]))
        return Point.from_name(move)
        

class TfBot(Bot):

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]])
        conv_layer1 = tf.layers.conv2d(self.X, 32, kernel_size=(5, 5), 
            padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv_layer1, 
            pool_size=[2, 2], strides=2)
        flat_layer = tf.layers.flatten(pool1)

        dnn_layer1 = tf.layers.dense(flat_layer, 1024, activation=tf.nn.relu)
        dnn_layer2 = tf.layers.dense(dnn_layer1, 128, activation=tf.nn.relu)
        action_pred = tf.nn.softmax(tf.layers.dense(dnn_layer2, output_size))

        Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
        advantages = tf.placeholder(tf.float32, name="reward_signal")

        log_lik = -Y * tf.log(action_pred)
        log_lik_adv = log_lik * advantages
        loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    

    def move(self, board, nth_move):
        state = np.array(board).reshape(1, 19, 19, 1)
        action_prob = self.sess.run(action_pred, feed_dict={self.X: state})
        action = np.random.choice(np.arange(19 * 19), p=action_prob[0])

        action_space = np.zeros(19 * 19)
        action_space[action] = 1

        x, y = np.where(action_space == 1.0)

        return x, y

        