import math

class EqualTempered:
    def __init__(self, octaves=3, start_freq=220, resolution=1000):
        self.octaves = octaves
        self.start_freq = start_freq
        self.resolution = resolution

        # Discrete units:
        self.units = self.octaves * 12 * self.resolution
        # print(self.units)
        self.factor = math.pow(2*self.octaves, 1/self.units)

    # x has to be between 0 an 1
    def from_0_1_to_f(self, x):
        dx = math.floor(x * (self.units))
        # print(dx)
        f = self.start_freq * math.pow(self.factor, dx)
        return f

if __name__ == '__main__':
    converter = EqualTempered(octaves=1, start_freq=440, resolution=1)
    print(converter.from_0_1_to_f(0))
    print(converter.from_0_1_to_f(0.25))
    print(converter.from_0_1_to_f(0.5))
    print(converter.from_0_1_to_f(0.75))
    print(converter.from_0_1_to_f(1))

    print("First 10:")
    r = [i/10 for i in range(10)]
    first10 = map(converter.from_0_1_to_f, r)
    for idx, note in enumerate(first10):
        print("Index: {}, Note: {}".format(idx, note))
    