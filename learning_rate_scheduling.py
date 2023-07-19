from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class SawtoothSchedule(LearningRateSchedule):
    def __init__(self, start_learning_rate=0.0001, end_learning_rate=0.000001, cycle_steps=100, random_fluctuation = 0.0, name=None):
        super(SawtoothSchedule, self).__init__()
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.cycle_steps = cycle_steps
        self.random_fluctuation = random_fluctuation
        self.name = name
    pass

    def __call__(self, step):
        phase = step % self.cycle_steps
        lr = self.start_learning_rate + (self.end_learning_rate-self.start_learning_rate)* (phase/self.cycle_steps)
        if (self.random_fluctuation>0):
            lr *= np.random.normal(1.0, self.random_fluctuation)
        return lr

    def get_config(self):
        return {
            "start_learning_rate": self.start_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "cycle_steps": self.cycle_steps,
            "random_fluctuation": self.random_fluctuation,
            "name": self.name
        }
