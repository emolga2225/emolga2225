#!/usr/bin/env python3
"""
Render individual channels from NSF (NES Sound Format) files.

Uses ctypes to call libgme (Game Music Emulator) directly.
No pip dependencies - just needs libgme.dll (Windows) or libgme.so (Linux).

Install libgme:
  Windows: Download libgme.dll from:
           https://github.com/ShiftMediaProject/game-music-emu/releases
           Place libgme.dll in the same folder as this script.
  Linux:   sudo apt install libgme-dev   (Ubuntu/Debian)
           sudo dnf install game-music-emu (Fedora)
  Mac:     brew install game-music-emu

NES channels rendered:
  Pulse 1    -> pulse1.wav   (melody)
  Pulse 2    -> pulse2.wav   (harmony)
  Triangle   -> triangle.wav (bass)
  Noise      -> noise.wav    (drums)
  All mixed  -> mix.wav      (full mix)
"""

import argparse
import ctypes
import ctypes.util
import platform
import struct
import sys
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# libgme ctypes loader
# ---------------------------------------------------------------------------

def _load_lib_windows():
    """Search for libgme.dll on Windows."""
    search_dirs = [
        Path(__file__).parent,          # Same dir as this script
        Path(sys.executable).parent,    # Python install dir
        Path('C:/Windows/System32'),
        Path('C:/Windows/SysWOW64'),
    ]
    for name in ('libgme.dll', 'gme.dll'):
        # Try PATH first
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
        # Then check specific directories
        for d in search_dirs:
            path = d / name
            if path.exists():
                try:
                    return ctypes.CDLL(str(path))
                except OSError:
                    pass
    return None


def _load_lib_unix():
    """Search for libgme on Linux/Mac."""
    names = {
        'Darwin': ['libgme.dylib', 'libgme.0.dylib', 'libgme.dylib'],
        'Linux':  ['libgme.so', 'libgme.so.0', 'libgme.so.1'],
    }.get(platform.system(), ['libgme.so'])

    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    found = ctypes.util.find_library('gme')
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            pass
    return None


def load_libgme():
    """Load libgme and set up function signatures. Raises RuntimeError if not found."""
    system = platform.system()
    lib = _load_lib_windows() if system == 'Windows' else _load_lib_unix()

    if lib is None:
        msg = [
            "",
            "ERROR: libgme not found.",
            "",
            "Install instructions:",
        ]
        if system == 'Windows':
            script_dir = Path(__file__).parent
            msg += [
                "  1. Go to: https://github.com/ShiftMediaProject/game-music-emu/releases",
                "  2. Download the latest release zip (e.g. libgme_MSVC17_x64.zip)",
                "  3. Extract libgme.dll and place it here:",
                f"       {script_dir / 'libgme.dll'}",
            ]
        elif system == 'Darwin':
            msg += ["  brew install game-music-emu"]
        else:
            msg += [
                "  Ubuntu/Debian: sudo apt install libgme-dev",
                "  Fedora:        sudo dnf install game-music-emu",
                "  Arch:          sudo pacman -S game-music-emu",
            ]
        msg.append("")
        raise RuntimeError("\n".join(msg))

    # --- gme_open_file(path, &emu, sample_rate) -> err ---
    lib.gme_open_file.restype = ctypes.c_char_p
    lib.gme_open_file.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]

    # --- gme_start_track(emu, index) -> err ---
    lib.gme_start_track.restype = ctypes.c_char_p
    lib.gme_start_track.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # --- gme_play(emu, count, short_buf) -> err ---
    lib.gme_play.restype = ctypes.c_char_p
    lib.gme_play.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_short),
    ]

    # --- gme_mute_voice(emu, voice_index, mute) ---
    lib.gme_mute_voice.restype = None
    lib.gme_mute_voice.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    # --- gme_voice_count(emu) -> int ---
    lib.gme_voice_count.restype = ctypes.c_int
    lib.gme_voice_count.argtypes = [ctypes.c_void_p]

    # --- gme_track_count(emu) -> int ---
    lib.gme_track_count.restype = ctypes.c_int
    lib.gme_track_count.argtypes = [ctypes.c_void_p]

    # --- gme_track_ended(emu) -> int ---
    lib.gme_track_ended.restype = ctypes.c_int
    lib.gme_track_ended.argtypes = [ctypes.c_void_p]

    # --- gme_delete(emu) ---
    lib.gme_delete.restype = None
    lib.gme_delete.argtypes = [ctypes.c_void_p]

    return lib


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CHUNK = 4096  # samples per render call (stereo = 2x shorts)


def render_channel(lib, nsf_path, output_path, track=0, duration=120,
                   sample_rate=48000, active_voices=None, fade_secs=4.0):
    """
    Render one channel configuration from an NSF file to a WAV file.

    Args:
        lib:           Loaded libgme ctypes handle.
        nsf_path:      Path to the NSF file.
        output_path:   Output WAV path.
        track:         0-based track index.
        duration:      Duration in seconds.
        sample_rate:   Output sample rate (Hz).
        active_voices: Set of voice indices to keep unmuted (None = all).
        fade_secs:     Fade-out duration at the end.
    """
    emu = ctypes.c_void_p(0)

    err = lib.gme_open_file(str(nsf_path).encode(), ctypes.byref(emu), sample_rate)
    if err:
        raise RuntimeError(f"gme_open_file: {err.decode()}")

    try:
        voice_count = lib.gme_voice_count(emu)

        # Mute voices not in the active set
        for i in range(voice_count):
            mute = 0 if (active_voices is None or i in active_voices) else 1
            lib.gme_mute_voice(emu, i, mute)

        err = lib.gme_start_track(emu, track)
        if err:
            raise RuntimeError(f"gme_start_track: {err.decode()}")

        # Render all stereo samples: duration * sample_rate * 2 (L+R)
        total_stereo = int(duration * sample_rate) * 2
        fade_stereo  = int(fade_secs * sample_rate) * 2
        buf = (ctypes.c_short * CHUNK)()
        samples = []

        rendered = 0
        while rendered < total_stereo:
            to_render = min(CHUNK, total_stereo - rendered)
            err = lib.gme_play(emu, to_render, buf)
            if err:
                raise RuntimeError(f"gme_play: {err.decode()}")
            samples.extend(buf[:to_render])
            rendered += to_render

        # Apply linear fade-out
        fade_start = max(0, len(samples) - fade_stereo)
        for i in range(fade_start, len(samples)):
            frac = (i - fade_start) / max(fade_stereo, 1)
            samples[i] = int(samples[i] * (1.0 - frac))

        # Write WAV
        audio_bytes = struct.pack(f'{len(samples)}h', *samples)
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

    finally:
        lib.gme_delete(emu)


def render_all_channels(nsf_path, output_dir, track=0, duration=120, sample_rate=48000):
    """
    Render all NES channels separately plus the full mix.

    NES voice indices (for NSF files via libgme):
        0 = Pulse 1
        1 = Pulse 2
        2 = Triangle
        3 = Noise
        4 = DMC (sample channel)
    """
    print("Loading libgme...")
    lib = load_libgme()
    print("[OK] libgme loaded\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nsf_name = Path(nsf_path).stem

    channels = [
        ('pulse1',   {0},        'Pulse 1 (melody)'),
        ('pulse2',   {1},        'Pulse 2 (harmony)'),
        ('triangle', {2},        'Triangle (bass)'),
        ('noise',    {3, 4},     'Noise + DMC (drums)'),
        ('mix',      None,       'Full Mix (all channels)'),
    ]

    print(f"NSF: {nsf_path}")
    print(f"Track: {track}, Duration: {duration}s, Sample rate: {sample_rate}Hz\n")

    for channel_name, voices, description in channels:
        output_path = output_dir / f"{nsf_name}_{channel_name}.wav"
        print(f"  Rendering {description}...")
        render_channel(
            lib, nsf_path, output_path,
            track=track,
            duration=duration,
            sample_rate=sample_rate,
            active_voices=voices,
        )
        print(f"  [OK] {output_path.name}")

    print(f"\n[DONE] All channels saved to: {output_dir}")
    print(f"\nStem mapping for training:")
    print(f"  pulse1   -> vocals")
    print(f"  pulse2   -> guitar")
    print(f"  triangle -> bass")
    print(f"  noise    -> drums")
    print(f"  mix      -> song")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Render individual NES channels from NSF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_nsf_channels.py game.nsf -o output/
  python render_nsf_channels.py game.nsf -o output/ --track 1 --duration 120
        """
    )
    parser.add_argument('nsf_file',               help='Path to NSF file')
    parser.add_argument('-o', '--output-dir',     default='nsf_channels',
                        help='Output directory (default: nsf_channels)')
    parser.add_argument('-t', '--track',  type=int, default=0,
                        help='Track number, 0-based (default: 0)')
    parser.add_argument('-d', '--duration', type=float, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('-r', '--sample-rate', type=int, default=48000,
                        help='Sample rate in Hz (default: 48000)')

    args = parser.parse_args()

    if not Path(args.nsf_file).exists():
        print(f"ERROR: NSF file not found: {args.nsf_file}")
        sys.exit(1)

    render_all_channels(
        args.nsf_file,
        args.output_dir,
        track=args.track,
        duration=args.duration,
        sample_rate=args.sample_rate,
    )


if __name__ == '__main__':
    main()
