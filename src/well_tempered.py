import math

class WellTempered:
    def __init__(self, octaves=3, start_freq=220, size=1000):
        self.octaves = octaves
        self.start_freq = start_freq
        self.size = size

        # Discrete units:
        self.units = self.octaves * 12 * self.size
        self.factor = math.pow(2, 1/self.units)

    # x has to be between 0 an 1
    def from_0_1_to_f(self, x):
        dx = math.floor(x * (self.units))
        print(dx)
        f = self.start_freq * math.pow(self.factor, dx)
        return f

if __name__ == '__main__':
    converter = WellTempered()
    print(converter.from_0_1_to_f(0))
    print(converter.from_0_1_to_f(0.5))
    print(converter.from_0_1_to_f(1))