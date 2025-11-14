import soundfile as sf
from hybrid_resolution_sinusoidal_extractor import HybridResolutionSinusoidalExtractor
import os

def batch_extract(files, export_individual_tracks=False, export_combined=False):
    """
    Batch process multiple audio files for sinusoidal extraction

    Args:
        files: List of audio file paths
        export_individual_tracks: Export each track as separate WAV
        export_combined: Export combined synthesis
    """
    print("=" * 60)
    print("BATCH SINUSOIDAL EXTRACTION")
    print("=" * 60)
    print(f"Files to process: {len(files)}")
    print(f"Export individual tracks: {export_individual_tracks}")
    print(f"Export combined synthesis: {export_combined}")
    print("=" * 60)

    for idx, audio_file in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(files)}: {audio_file}")
        print(f"{'='*60}")

        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            continue

        try:
            # Detect sample rate
            print(f"📂 Reading: {audio_file}")
            _, sr_detected = sf.read(audio_file, frames=1)
            print(f"   Detected sample rate: {sr_detected} Hz")

            # Create extractor with detected sample rate
            extractor = HybridResolutionSinusoidalExtractor(sample_rate=sr_detected)

            # Analyze
            original, all_channel_tracks, is_stereo = extractor.analyze(audio_file)

            print(f"\n✓ Band extraction and analysis complete")

            # Save tracks to HDF5
            base_name = os.path.splitext(audio_file)[0]
            hdf5_file = f"{base_name}_tracks.h5"
            extractor.save_to_hdf5(hdf5_file, all_channel_tracks, is_stereo, audio_file)

            n_samples = len(original)

            # Export individual tracks as WAV files (optional)
            if export_individual_tracks:
                track_dir = f"{base_name}_tracks"
                extractor.export_tracks_as_wav(all_channel_tracks, n_samples, output_dir=track_dir)

            # Synthesize and export combined audio (optional)
            if export_combined:
                synthesized = extractor.synthesize_stereo(all_channel_tracks, n_samples, is_stereo)
                output_file = f"{base_name}_synthesized.wav"
                sf.write(output_file, synthesized, extractor.sample_rate)
                print(f"\n✓ Exported synthesized audio: {output_file}")

            print(f"\n✅ SUCCESS: {audio_file}")

        except Exception as e:
            print(f"\n❌ ERROR processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Guitar Hero files
    files = [
        "bass.ogg",
        "drums_1.ogg",
        "drums_2.ogg",
        "drums_3.ogg",
        "drums_4.ogg",
        "fullmix.ogg",
        "guitar.ogg",
        "song.ogg",
        "vocals.ogg"
    ]

    # Configuration
    export_individual_tracks = False  # Set to True to export each track as separate WAV
    export_combined = False           # Set to True to export combined synthesis

    # Run batch extraction
    batch_extract(files, export_individual_tracks, export_combined)
