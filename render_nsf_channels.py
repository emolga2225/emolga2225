#!/usr/bin/env python3
"""
Render individual channels from NSF (NES Sound Format) files.

This script extracts each NES audio channel as a separate WAV file:
- Pulse 1 (melody)
- Pulse 2 (harmony)
- Triangle (bass)
- Noise (drums/percussion)
- Mix (all channels combined)

Requires: pip install game-music-emu
"""

import argparse
import struct
import wave
from pathlib import Path

try:
    import gme
except ImportError:
    print("ERROR: game-music-emu not installed")
    print("Install with: pip install game-music-emu")
    exit(1)


def render_nsf_channel(nsf_path, output_path, track=0, duration=180, sample_rate=48000,
                       channel_mask=0x1F, fade_length=8.0):
    """
    Render a specific channel configuration from an NSF file.

    Args:
        nsf_path: Path to NSF file
        output_path: Path to output WAV file
        track: Track number (default 0)
        duration: Duration in seconds
        sample_rate: Sample rate (default 48000 Hz)
        channel_mask: Bitmask for which channels to enable
                      Bit 0: Pulse 1
                      Bit 1: Pulse 2
                      Bit 2: Triangle
                      Bit 3: Noise
                      Bit 4: DMC
                      0x1F = all channels (00011111)
        fade_length: Fade out length in seconds
    """
    # Load NSF file
    music_emu = gme.MusicEmu(str(nsf_path), sample_rate)
    music_emu.track = track

    # Set channel muting based on mask
    # GME uses voice muting - we mute channels NOT in the mask
    for i in range(5):  # NES has 5 channels
        should_mute = not (channel_mask & (1 << i))
        music_emu.mute_voice(i, should_mute)

    # Start playback
    music_emu.start_track()

    # Calculate sample count
    num_samples = int(duration * sample_rate)
    fade_samples = int(fade_length * sample_rate)

    # Render audio
    print(f"  Rendering {duration}s at {sample_rate}Hz...")
    samples = music_emu.play(num_samples)

    # Convert from interleaved stereo shorts to bytes
    # GME returns array of 16-bit signed integers (stereo interleaved)
    audio_data = struct.pack(f'{len(samples)}h', *samples)

    # Apply fade out
    if fade_samples > 0 and num_samples > fade_samples:
        # Convert back to list for fade processing
        sample_list = list(samples)
        fade_start = (num_samples - fade_samples) * 2  # *2 for stereo

        for i in range(fade_samples):
            fade_factor = 1.0 - (i / fade_samples)
            # Apply to both left and right channels
            idx = fade_start + (i * 2)
            sample_list[idx] = int(sample_list[idx] * fade_factor)
            sample_list[idx + 1] = int(sample_list[idx + 1] * fade_factor)

        audio_data = struct.pack(f'{len(sample_list)}h', *sample_list)

    # Write WAV file
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    print(f"  Wrote {output_path}")


def render_all_channels(nsf_path, output_dir, track=0, duration=180, sample_rate=48000):
    """
    Render all NES channels separately plus the full mix.

    Creates 5 files:
        pulse1.wav    - Pulse wave 1 (melody)
        pulse2.wav    - Pulse wave 2 (harmony)
        triangle.wav  - Triangle wave (bass)
        noise.wav     - Noise channel (drums)
        mix.wav       - All channels combined
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nsf_name = Path(nsf_path).stem

    channels = [
        ('pulse1', 0x01, 'Pulse 1'),
        ('pulse2', 0x02, 'Pulse 2'),
        ('triangle', 0x04, 'Triangle'),
        ('noise', 0x08, 'Noise'),
        ('mix', 0x1F, 'Full Mix'),
    ]

    print(f"Rendering NSF: {nsf_path}")
    print(f"Track: {track}, Duration: {duration}s, Sample rate: {sample_rate}Hz\n")

    for channel_name, mask, description in channels:
        output_path = output_dir / f"{nsf_name}_{channel_name}.wav"
        print(f"{description}:")
        render_nsf_channel(
            nsf_path,
            output_path,
            track=track,
            duration=duration,
            sample_rate=sample_rate,
            channel_mask=mask
        )
        print()

    print(f"[DONE] All channels rendered to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Extract sinusoidal features:")
    print(f"   for file in {output_dir}/*.wav; do")
    print(f"     python ssq_sinusoidal_extractor.py \"$file\"")
    print(f"   done")
    print(f"\n2. Create training data with the 5 HDF5 files (pulse1, pulse2, triangle, noise, mix)")


def main():
    parser = argparse.ArgumentParser(
        description='Render individual NES channels from NSF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render all channels from track 0 (default 3 minutes)
  python render_nsf_channels.py game.nsf -o output/

  # Render specific track for 2 minutes
  python render_nsf_channels.py game.nsf -o output/ --track 1 --duration 120

  # Custom sample rate
  python render_nsf_channels.py game.nsf -o output/ --sample-rate 48000
        """
    )

    parser.add_argument('nsf_file', type=str, help='Path to NSF file')
    parser.add_argument('-o', '--output-dir', type=str, default='nsf_channels',
                        help='Output directory (default: nsf_channels)')
    parser.add_argument('-t', '--track', type=int, default=0,
                        help='Track number to render (default: 0)')
    parser.add_argument('-d', '--duration', type=float, default=180,
                        help='Duration in seconds (default: 180)')
    parser.add_argument('-r', '--sample-rate', type=int, default=48000,
                        help='Sample rate in Hz (default: 48000)')

    args = parser.parse_args()

    if not Path(args.nsf_file).exists():
        print(f"ERROR: NSF file not found: {args.nsf_file}")
        exit(1)

    render_all_channels(
        args.nsf_file,
        args.output_dir,
        track=args.track,
        duration=args.duration,
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()
