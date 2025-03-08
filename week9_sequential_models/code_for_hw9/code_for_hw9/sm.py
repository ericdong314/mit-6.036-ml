from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        outputs = []
        state = self.start_state
        for input in input_seq:
            state = self.transition_fn(state, input)
            outputs.append(self.output_fn(state))
        return outputs

class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0) # Change

    def transition_fn(self, s, x):
        # Your code here
        digit, carry = s
        total = carry + x[0] + x[1]
        return total % 2, total // 2

    def output_fn(self, s):
        # Your code here
        return s[0]


class Reverser(SM):
    start_state = ([None], 0)

    def transition_fn(self, s, x):
        # Your code here
        words, end_seen = s
        words = words[:-1] + [x, None]
        if x == 'end':
            end_seen = 1
        if end_seen:
            words.pop()
            words.pop()
        return words, end_seen

    def output_fn(self, s):
        # Your code here
        return s[0][-1] if s[0] else None


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2

        self.start_state = np.zeros((self.Wo.shape[1], 1))

    def transition_fn(self, s, x):
        # Your code here
        return self.f1(self.Wss @ s + np.dot(self.Wsx, x) + self.Wss_0)

    def output_fn(self, s):
        # Your code here
        return self.f2(self.Wo @ s + self.Wo_0)
