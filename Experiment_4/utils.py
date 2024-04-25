# Counter of forward and backward passes.
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        return self.val
