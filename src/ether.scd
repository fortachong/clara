// Start Server
s.boot;

// Theremin Model I
(
SynthDef.new(\ether, {
	arg frequency=440, amplitude=0.01;

	var f0, f1, signal;
	f0 = 50000;
	f1 = f0 + frequency;
	signal = BPF.ar(
		SinOsc.ar(f0) * SinOsc.ar(f1), abs(f1-f0), 0.2
	) * amplitude;
	Out.ar(0, signal);
	Out.ar(1, signal);
}).add;
);

(
// Instantiate
y = Synth.new(\ether);
// OSC Frequency control
f = OSCFunc({ | msg, time, addr, port |
	var pyFreq;

	pyFreq = msg[1].asFloat;
	( "freq is " + pyFreq ).postln;
	y.set( \frequency, pyFreq);
}, '/freq');
// OSC Amplitude control
a = OSCFunc( { | msg, time, addr, port |
	var pyAmp;

	pyAmp = msg[1].asFloat;
	( "amp is " + pyAmp ).postln;
	y.set( \amplitude, pyAmp );
}, '/amp');
);