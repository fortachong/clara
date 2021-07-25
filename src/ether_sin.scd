// Server.killAll;
// Start Server
s.boot;
NetAddr.langPort;

// Theremin Model I
(
SynthDef.new(\ether, {
	arg frequency=440, amplitude=0;

	var signal, max_amplitude;
	max_amplitude = 0.5;
	SinOsc.ar(frequency) * amplitude * max_amplitude;
	signal.scope;
	Out.ar(0, signal);
	Out.ar(1, signal);
}).add;
);

(
// Instantiate
y = Synth.new(\ether);
FreqScope.new(server: s);
// OSC Frequency control
OSCdef.new(
	\freq,
	{
		arg msg, time, addr, port;
		var pyFreq;
		[msg, time, addr, port].postln;
		pyFreq = msg[1].asFloat;
		( "freq is " + pyFreq ).postln;
		y.set( \frequency, pyFreq);
	},
	'/main/f'
);

// OSC Amplitude control
OSCdef.new(
	\amp,
	{
		arg msg, time, addr, port;
		var pyAmp;
		[msg, time, addr, port].postln;
		pyAmp = msg[1].asFloat;
		( "amp is " + pyAmp ).postln;
		y.set( \amplitude, pyAmp);
	},
	'/main/a'
);

);