"""
Synthesised F1-style engine audio (no recordings needed).

The engine note is built from the firing frequency of a 6-cylinder 4-stroke
(rpm / 60 * 3) plus harmonics, soft-clipped for grit, with gear shifts and a
Doppler effect for the fly-by.
"""
import io
import wave

import numpy as np

SR = 22050


def _to_wav(signal):
    signal = signal / (np.abs(signal).max() + 1e-9) * 0.9
    pcm = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _engine(rpm, seed=0):
    """Turn an rpm-per-sample curve into an engine tone."""
    rng = np.random.default_rng(seed)
    f0 = rpm / 60 * 3  # V6 firing frequency
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = np.zeros_like(rpm)
    for h, amp in [(0.5, 0.5), (1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2), (6, 0.12)]:
        tone += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    # combustion roughness: noise pulsing with each firing
    tone += 0.25 * rng.normal(size=len(rpm)) * (0.5 + 0.5 * np.sin(phase))
    tone = np.tanh(1.8 * tone)  # soft clip
    # turbo whistle rising with revs
    tone += 0.04 * np.sin(2 * np.pi * np.cumsum(2.2 * f0 + 2500) / SR)
    return tone


def _fade(sig, ms_in=20, ms_out=300):
    n_in, n_out = int(SR * ms_in / 1000), int(SR * ms_out / 1000)
    sig[:n_in] *= np.linspace(0, 1, n_in)
    sig[-n_out:] *= np.linspace(1, 0, n_out)
    return sig


def race_start():
    """Five red lights, idle revs, lights out, launch and up-shifts."""
    parts = []
    # countdown: idle engine with five beeps a second apart
    idle_len = 5.6
    t = np.arange(int(SR * idle_len)) / SR
    idle_rpm = 6500 + 900 * np.abs(np.sin(2 * np.pi * 0.7 * t)) ** 3  # driver blipping throttle
    idle = 0.45 * _engine(idle_rpm, seed=1)
    for i in range(5):
        start = int(SR * (0.4 + i))
        n = int(SR * 0.18)
        bt = np.arange(n) / SR
        idle[start:start + n] += 0.6 * np.sin(2 * np.pi * 880 * bt) * np.exp(-bt * 8)
    parts.append(idle)

    # lights out: launch through gears 1-5
    rpm = []
    lo, hi = 9500, 12000
    for gear, secs in enumerate([0.9, 1.0, 1.15, 1.3, 1.5]):
        n = int(SR * secs)
        start_rpm = 7500 if gear == 0 else lo
        rpm.append(start_rpm + (hi - start_rpm) * (1 - np.exp(-np.linspace(0, 3, n))) / (1 - np.exp(-3)))
        rpm.append(np.linspace(hi, lo, int(SR * 0.04)))  # seamless shift dip
    rpm = np.concatenate(rpm)
    launch = _engine(rpm, seed=2)
    launch *= np.linspace(1.0, 0.55, len(launch))  # car pulling away
    parts.append(launch)

    return _to_wav(_fade(np.concatenate(parts)))


def fly_by(speed_kmh=320, duration=4.0):
    """A car passing the listener at speed, with Doppler shift and distance fade."""
    n = int(SR * duration)
    t = np.arange(n) / SR - duration / 2  # car passes at t = 0
    v = speed_kmh / 3.6
    x, d = v * t, 8.0  # position along track; listener 8 m from racing line
    dist = np.sqrt(x ** 2 + d ** 2)
    radial_v = v * x / dist  # +ve when moving away
    doppler = 343 / (343 + radial_v)
    rpm = np.full(n, 11800.0) * doppler
    sig = _engine(rpm, seed=3) / (1 + (dist / 12) ** 1.6)
    return _to_wav(_fade(sig, 50, 400))


if __name__ == "__main__":
    for name, fn in [("race_start.wav", race_start), ("fly_by.wav", fly_by)]:
        with open(name, "wb") as fh:
            fh.write(fn())
        print("wrote", name)
