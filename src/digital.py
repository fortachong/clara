import math

# Simulation of Theremin Digital Heterodyne

# Please see:
# http://www.thereminworld.com/Forums/T/30562/lets-design-and-build-a-simple-analog-theremin
# "In the DIGITAL the period of the result of heterodyning is measured via a 
# system clock then numerically offset"

# Theremin Parameters
TH_PARAMS = {
    'ANTENNA_LENGTH': 250,      # Antenna Length in mm
    'ANTENNA_DIAMETER': 10,     # Antenna Diameter in mm
    'L_TANK': 0.6,              # Inductance in LC Tank in mH
    'C_TANK': 0,                # Capacitance in LC Tank in pF
    'C_BLOCK': 100,             # Block Capacitance in pF
    'C_STRAY': 3,               # Stray Capacitance in pF
    'HAND_DIMENSION': 0.18,     # Hand dimension in mts
    'FIDDLE': 8.2,              # Fiddle
    'EPSILON0': 8.85e-12,       # In Fm^-1
    'HAND_INTERVAL': 30,        # Hand interval in mm
    'K': 0.4,                   # K constant
    'SCALE_SIZE': 31,           # How many values in the table
    'SYSTEM_CLOCK': 160,        # System Clock in MHz
    'FAR_FIELD_NULL': 25,       # Frequency of Far Field Null in kHz
    'NUMERIC_OFFSET': 50        # Numeric Offset
}

class HeterodyneDigital:
    def __init__(self, params=TH_PARAMS):
        self.params = params
        print("Initializing Heterodyne (Digital) Parameters...")
        # Antenna Length in meters
        self.antenna_length = self.params['ANTENNA_LENGTH']/1000
        print("Antenna Length: {} mts".format(self.antenna_length))
        # Antenna Diameter in meters
        self.antenna_diameter = self.params['ANTENNA_DIAMETER']/1000
        print("Antenna Diameter: {} mts".format(self.antenna_diameter))
        # Tank Inductance in Henris
        self.l_tank = self.params['L_TANK']/1000
        print("Tank Inductance: {} H".format(self.l_tank))
        # Tank Capacitance in Farads
        self.c_tank = self.params['C_TANK']*(1e-12)
        print("Tank Capacitance: {} F".format(self.c_tank))
        # Block Capacitance in Farads
        self.c_block = self.params['C_BLOCK']*(1e-12)
        self.inv_c_block = 1/self.c_block
        print("Block Capacitance: {} F".format(self.c_block))
        # Stray Capacitance in Farads
        self.c_stray = self.params['C_STRAY']*(1e-12)
        print("Stray Capacitance: {} F".format(self.c_stray))
        # Hand Dimension
        self.hand_dimension = self.params['HAND_DIMENSION']
        print("Hand Dimension: {} mts".format(self.hand_dimension))
        # Fiddle
        self.fiddle = self.params['FIDDLE']
        print("Fiddle: {}".format(self.fiddle))
        # Epsilon constant
        self.epsilon0 = self.params['EPSILON0']
        print("Epsilon0: {} Fm^-1".format(self.epsilon0))
        # Hand Interval
        self.hand_interval = self.params['HAND_INTERVAL']
        print("Hand Interval: {} mm".format(self.hand_interval))
        # K Factor
        self.k = self.params['K']
        print("K Factor: {}".format(self.k))
        # Scale Size
        self.scale_size = self.params['SCALE_SIZE']
        print("Table Size: {}".format(self.scale_size))
        # System Clock
        self.system_clock = self.params['SYSTEM_CLOCK'] * 1000000
        print("System Clock: {} Hz".format(self.system_clock))
        # Far field Null Frequency
        self.far_field_null = self.params['FAR_FIELD_NULL'] * 1000
        print("Far Field Null Frequency: {} Hz".format(self.far_field_null))
        # Numeric offset
        self.numeric_offset = self.params['NUMERIC_OFFSET']
        print("Numeric Offset: {}".format(self.numeric_offset))
        # Far count
        self.far_count = self.numeric_offset + self.system_clock/self.far_field_null
        
        self.factor1 = self.fiddle*self.epsilon0
        self.factor2 = math.sqrt(self.antenna_length*self.hand_dimension)/math.pi

        # Antenna Capacitance
        self.c_antenna = 2*math.pi*self.epsilon0*self.antenna_length/(math.log(2*self.antenna_length/self.antenna_diameter) - self.k)
        print("Antenna Capacitance c_antenna: {} F".format(self.c_antenna))
        
        # Generates a table 
        self.table_idxs = range(self.scale_size)
        self.distances = [self.hand_interval * (i+1) for i in self.table_idxs]
        self.hand_capacitances = list(map(self.calculate_hand_capacitance, self.distances))
        self.total_capacitances = list(map(self.calculate_total_capacitance, self.distances))
        self.frequencies = list(map(self.calculate_frequency, self.distances))
        print("Frequency Table:")
        print("Indexes:")
        print(list(self.table_idxs))
        print("Distance:")
        print(self.distances)
        print("Hand Capacitance:")
        print(self.hand_capacitances)
        print("Total Capacitance:")
        print(self.total_capacitances)
        print("Frequency:")
        print(self.frequencies)

        # Fixed Oscillator
        self.fixed_osc_frequency = self.frequencies[-1] + self.far_field_null
        print("Fixed Oscilator at: {}".format(self.fixed_osc_frequency))

        # Beats
        self.beats = list(map(self.calculate_beat, self.distances))
        print("Beat Frequency:")
        print(self.beats)

        # Counts
        self.counts = list(map(self.calculate_count, self.distances))
        print("Count:")
        print(self.counts)

        # Offsets
        self.offsets = list(map(self.calculate_offset, self.distances))
        print("Offset:")
        print(self.offsets)

    # Calculate Hand Capacitance, d in mm
    def calculate_hand_capacitance(self, d):
        x = d/1000
        f1 = x + self.antenna_diameter
        f2 = x + self.hand_dimension
        f3 = x*(f1 + self.hand_dimension)
        f4 = math.log(f1*f2/f3)
        c_hand = self.factor1 * self.factor2 * f4
        return c_hand

    # Calculate Total Capacitance, d in mm
    def calculate_total_capacitance(self, d):
        c_hand = self.calculate_hand_capacitance(d)
        i1 = 1/(self.c_stray+self.c_antenna+c_hand)
        c_total = self.c_tank + 1/(i1+self.inv_c_block)
        return c_total

    # Frequency, d in mm
    def calculate_frequency(self, d):
        c_total = self.calculate_total_capacitance(d)
        freq = 1/(2*math.pi*math.sqrt(c_total*self.l_tank))
        return freq

    # Beat, d in mm
    def calculate_beat(self, d):
        f1 = self.calculate_frequency(d)
        beat = abs(f1 - self.fixed_osc_frequency)
        return beat

    # Count, d in mm
    def calculate_count(self, d):
        c1 = self.calculate_beat(d)
        count = self.system_clock / c1
        return count

    # Offset, d in mm
    def calculate_offset(self, d):
        o1 = self.calculate_count(d)
        offset = self.far_count - o1
        return offset

    # No yet implemented
    def generate_scale(self):
        pass

if __name__ == '__main__':
    het = HeterodyneDigital()
    print("Frequency at 500mm:")
    f1 = het.calculate_offset(500)
    print("{} Hz".format(f1))
    print()
    print("Frequency at 250mm:")
    f2 = het.calculate_offset(250)
    print("{} Hz".format(f2))

        