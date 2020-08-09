import numpy as np
from hmmlearn import hmm

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

    def observation_to_index(self, o_list):
        # 将观察序列转换为index list
        o_index = []
        for o in o_list:
            o_index.append(self.emission.index(o))
        return o_index

    def index_to_state(self, s_index):
        # 将index list转换为状态序列
        s_list= []
        for i in s_index:
            s_list.append(self.station[i])
        return s_list


def Viterbi(hmm, o_index):
    """
    :param hmm:HMM参数
    :param o_index:观测序列的index
    :return:最有可能的状态序列
    """

    T = len(o_index)   # length of the observation sequence

    # the size of all delta and psi is T*N
    delta = np.zeros((T, hmm.n))
    psi = np.zeros((T, hmm.n), dtype=np.int16)

    # initialize, t=0
    for i in range(hmm.n):
        delta[0][i] = hmm.Pi[i] * hmm.B[i][o_index[0]]

    # from t=1 to t=T calculate delta and psi
    for t in range(1,T):
        # calculate all hidden states probability
        for i in range(hmm.n):
            max_proba = 0.0
            max_index = 0
            for j in range(hmm.n):
                proba = delta[t-1][j]*hmm.A[j][i]
                if proba>max_proba:
                    max_proba = proba
                    max_index = j
            delta[t][i] = max_proba * hmm.B[i][o_index[t]]
            psi[t][i] = max_index

    # print(delta)
    # print(psi)

    # backtracking to find the path
    s_index = np.zeros(T, dtype=np.int16)
    s_index[T-1] = delta[T - 1, :].argmax()
    for t in range(T-2,-1,-1):
        s_index[t] = psi[t+1][s_index[t+1]]

    return hmm.index_to_state(s_index)


if __name__ == "__main__":

    n = 2   # number of hidden states
    m = 4   # number of emission state
    station = ["q1", "q2"]
    emission = ["v1", "v2", "v3", "v4"]

    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.3, 0.2, 0.4], [0.5, 0.1, 0.1, 0.2]])
    Pi = np.array([0.7, 0.3])

    my_hmm = HMM(n, m, station, emission, A, B, Pi)
    observation = ["v3", "v1", "v4", "v1", "v2"]
    o_index = my_hmm.observation_to_index(observation)
    s_list = Viterbi(my_hmm, o_index)
    print("My HMM model:", ", ".join(s_list))

    # 使用hmmlearn库进行验证
    # 通过命令 $ pip install hmmlearn安装
    model = hmm.MultinomialHMM(n_components=n)
    model.startprob_ = Pi
    model.transmat_ = A
    model.emissionprob_ = B
    o = np.array([o_index]).T
    logprob, h = model.decode(o, algorithm="viterbi")
    print("The hmmlearn:", ", ".join(map(lambda x: station[x], h)))
