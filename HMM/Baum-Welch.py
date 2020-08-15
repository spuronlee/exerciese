
import numpy as np
from hmmlearn import hmm
import random

def draw_from(probs):
    return np.where(np.random.multinomial(1, probs) == 1)[0][0]


# HMM类，用于存储HMM的参数
class HMM(object):
    def __init__(self, n, m, station, emission, A, B, Pi):
        self.n = n     # number of hidden states
        self.m = m     # number of emission state
        self.station = station    # hidden sates list
        self.emission = emission  # emission sates list
        self.A = A     # transition probability
        self.B = B     # emission probability
        self.Pi = Pi   # start state probability

        # 利用断言进行参数检验
        assert self.n == len(self.station)
        assert self.m == len(self.emission)
        assert self.A.size == self.n*self.n
        assert self.B.size == self.m*self.n

    def simulate(self, T):
        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.Pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t - 1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states

    def obsSeq_to_obsIndex(self, o_list):
        # 将观察序列转换为index list
        o_index = []
        for o in o_list:
            o_index.append(self.emission.index(o))
        return o_index

    def staIindex_to_staSeq(self, s_index):
        # 将状态的index list转换为状态序列
        s_list= []
        for i in s_index:
            s_list.append(self.station[i])
        return s_list

    def _forward(self, obs_seq):
        T = len(obs_seq)               # 序列长度
        alpha = np.zeros((self.n, T))  # 初始化
        alpha[:, 0] = self.Pi * self.B[:, obs_seq[0]]  # 公式(5.15)
        for t in range(1, T):    # 公式(5.16)
            for n in range(self.n):
                alpha[n, t] = np.dot(alpha[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]
        return alpha

    def _backward(self, obs_seq):
        T = len(obs_seq)              # 序列长度
        beta = np.zeros((self.n, T))  # 初始化
        beta[:, -1:] = 1              # 公式(5.19)
        for t in reversed(range(T - 1)):  # 公式(5.20)
            for n in range(self.n):
                beta[n, t] = sum(beta[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])
        return beta

    def baum_welch_train(self, obs_seq, iter_times=3):

        T = len(obs_seq)              # 观察序列的长度T
        for _ in range(iter_times):   # 迭代次数
            # Initialize alpha
            alpha = self._forward(obs_seq)
            # Initialize beta
            beta = self._backward(obs_seq)
            # calculate gamma
            gamma = np.zeros((self.n, T))
            for t in range(T):
                denominator = 0
                for i in range(self.n):
                    denominator += alpha[i][t] * beta[i][t]
                for i in range(self.n):
                    gamma[i][t] = alpha[i][t] * beta[i][t] / denominator

            # calculate xi(i,j,t)
            xi = np.zeros((self.n, self.n, T - 1))
            for t in range(T-1):
                denominator = 0
                for i in range(self.n):
                    for j in range(self.n):
                        denominator += alpha[i][t] * self.A[i][j] * self.B[j][obs_seq[t+1]] * beta[j][t+1]
                for i in range(self.n):
                    for j in range(self.n):
                        xi[i][j][t] = alpha[i][t] * self.A[i][j] * self.B[j][obs_seq[t+1]] * beta[j][t+1] / denominator

            # update
            newpi = gamma[:, 0]                                                     # 公式(5.26)
            newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))   # 公式(5.24)
            newB = np.copy(self.B)                                                  # 公式(5.25)
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(self.m):
                mask = obs_seq == lev
                newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma
            self.A, self.B, self.Pi = newA, newB, newpi


if __name__ == "__main__":

    # 运行参数
    iter_times = 6     # baum_welch算法迭代次数
    sample_low = 400   # 采样的范围下界
    sample_high = 500  # 采样的范围上界
    epoch = 100        # 训练次数

    # HMM参数，用来产生模拟数据（观测序列）
    n = 2   # number of hidden states
    m = 4   # number of emission state
    station = ["q1", "q2"]
    emission = ["v1", "v2", "v3", "v4"]
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.3, 0.2, 0.4], [0.5, 0.1, 0.1, 0.2]])
    Pi = np.array([0.7, 0.3])
    origin_hmm = HMM(n, m, station, emission, A, B, Pi)

    # 初始化baum_welch算法
    a = np.array([[0.49, 0.51], [0.52, 0.48]])
    b = np.array([[0.254, 0.154, 0.332, 0.260], [0.2, 0.3, 0.2, 0.3]])
    pi = np.array([0.394, 0.606])
    estimate_hmm = HMM(n, m, station, emission, a, b, pi)

    min_pi = 100
    min_a = 100
    min_b = 100
    for e in range(epoch):
        # 产生观测序列的长度
        T = np.random.randint(sample_low, sample_high)
        # 根据给定的hmm参数生成模拟数据
        obs_seq, _ = origin_hmm.simulate(T)
        # baum welch算法
        estimate_hmm.baum_welch_train(obs_seq, iter_times)
        pi_differ = np.sum(abs(Pi - estimate_hmm.Pi))
        a_differ = np.sum(abs(A - estimate_hmm.A))
        b_differ = np.sum(abs(B - estimate_hmm.B))
        if pi_differ<min_pi and a_differ<min_a and b_differ<min_b:
            min_pi = pi_differ
            min_a = a_differ
            min_b = b_differ
            print("-------------------------------------------")
            print("epoch:"+str(e) + ",   " + str(pi_differ) + "," + str(a_differ) + "," + str(b_differ))
            print(estimate_hmm.A)
            print(estimate_hmm.B)
            print(estimate_hmm.Pi)


    # n = 3   # number of hidden states
    # m = 2   # number of emission state
    # station = ["1", "2", "3"]
    # emission = ["r", "w"]
    #
    # A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    # Pi = np.array([0.2, 0.4, 0.4])
    # origin_hmm = HMM(n, m, station, emission, A, B, Pi)
    # o = origin_hmm.obsSeq_to_obsIndex(["r", "w", "r"])
    # origin_hmm.baum_welch_train(o)

    # a = np.array([[0.8, 0.1, 0.1], [0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
    # b = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
    # pi = np.array([0.6, 0.2, 0.2])
    # estimate_hmm = HMM(n, m, station, emission, a, b, pi)
    #
    #
    # min_pi = 10
    # min_a = 10
    # min_b = 10
    # for iter in range(1000):
    #     T = np.random.randint(50, 200)
    #     obs_seq, _ = origin_hmm.simulate(T)
    #     estimate_hmm.baum_welch_train(obs_seq)
    #     pi_differ = np.sum(abs(Pi - estimate_hmm.Pi))
    #     a_differ = np.sum(abs(A - estimate_hmm.A))
    #     b_differ = np.sum(abs(B - estimate_hmm.B))
    #     if pi_differ<min_pi and a_differ<min_a and b_differ<min_b:
    #         min_pi = pi_differ
    #         min_a = a_differ
    #         min_b = b_differ
    #         print("-------------"+str(iter)+"----------------------")
    #         print(str(pi_differ) + "," + str(a_differ) + "," + str(b_differ))
    #         print(estimate_hmm.A)
    #         print(estimate_hmm.B)
    #         print(estimate_hmm.Pi)
