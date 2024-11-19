import textwrap
from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from pydub import AudioSegment
import argparse
import os
import time

__doc__ = """
Transcription script using OpenAI API
=====================================
The audio file is split into 10-minute parts and transcribed using the Whisper model.
The transcriptions are then combined into a single VTT or TXT file.

Important Note about OpenAI API Key:
-------------------------------
You need to have an OpenAI API key to use this script. You can get an API key by signing up at https://platform.openai.com/signup
You can set the API key using the OPENAI_API_KEY environment variable or pass it as an argument to the script using the -k flag.
To set an environment variable, you can use the following command:
Linux : export OPENAI_API_KEY=<your_openai_api_key>
Windows : set OPENAI_API_KEY=<your_openai_api_key>
"""


def split_file(file_name: str, split_size: int = 10) -> str:
    """
        Split Files into subparts
    Parameters:
        file_name (str) : Path to audio file to process
        split_size (int) : length of audio split in minutes (default = 10)
    """
    basename: str = os.path.basename(file_name).split(".")[0]
    source_audio = AudioSegment.from_mp3(file_name)

    # Define time in milliseconds for a 10-minute slice
    slice_duration: int = split_size * 60 * 1000  # 10 minutes in milliseconds
    total_duration: int = len(source_audio)

    # Create directory for output files if it doesn't exist
    output_dir: str = os.path.join('slices', basename)
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Splitting {file_name} in parts of {split_size} minutes in the {output_dir} directory...")

    # Iterate over the audio in split_size-minute intervals
    index: int = 0
    for start_time in range(0, total_duration, slice_duration):
        # Avoid going beyond the end
        end_time: int = min(start_time + slice_duration, total_duration)
        clipped_audio = source_audio[start_time:end_time]
        output_file: str = os.path.join(
            output_dir, f"{basename}_{index}_{start_time}_{end_time}.mp3")
        if not os.path.exists(output_file):
            clipped_audio.export(
                output_file,
                format="mp3"
            )
        else:
            print(f"File {output_file} already exists. Skipping...")
        index += 1
    print(f"Splitting {file_name} completed.")
    return output_dir


def get_transcript(audio_dir: str, openai_api_key: str, split_size: int, output_format: str = "vtt") -> str:
    """
        Get Transcription from OpenAI API
        general use documentation : https://platform.openai.com/docs/guides/speech-to-text 
        API reference : https://platform.openai.com/docs/api-reference/audio
    Parameters:
        audio_dir (str) : Path to audio file to process
        openai_api_key (str) : OpenAI API key
    """
    if openai_api_key != "":
        client = OpenAI(api_key=openai_api_key)
        print("Using provided OpenAI API key")
    else:
        print("Using OpenAI API key from the `OPENAI_API_KEY` environment variable")
        client = OpenAI()

    audio_files: list[str] = os.listdir(audio_dir)
    transcription = ""
    for index, audio_file in enumerate(audio_files):
        print(f"Transcribing [{index + 1}/{len(audio_files)}] {audio_file}...")
        file_path: str = os.path.join(audio_dir, audio_file)
        audio_file = open(file_path, "rb")
        file_transcription: TranscriptionVerbose = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            language="fr",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        if output_format == "vtt":
            transcription += convert_to_vtt(file_transcription,
                                            index * split_size)
        else:
            transcription += file_transcription.text
    return transcription


def seconds_to_vtt_time(seconds) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"


def convert_to_vtt(transcription_result: TranscriptionVerbose, offset: int = 0) -> str:
    transcription_dict = transcription_result.to_dict()
    segments = transcription_dict.get("segments", [])
    offset_seconds: int = offset * 60
    if offset == 0:
        vtt_output = "WEBVTT\n\n"
    else:
        vtt_output = ""
    # Convert each segment to VTT format with the offset
    for segment in segments:  # type: ignore
        start = segment["start"] + offset_seconds
        end = segment["end"] + offset_seconds
        text = segment["text"]
        start_time: str = seconds_to_vtt_time(start)
        end_time: str = seconds_to_vtt_time(end)
        # Append each segment to the VTT output
        vtt_output += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_output


def main(file_name: str, split_size: int = 10, format: str = "vtt", openai_api_key: str = "", only_split: bool = False) -> None:
    """
        Process audio file to be transcripted using OpenAI Whisper API
    Parameters:
        file_name (str) : Path to audio file to process
        split_size (int) : length of audio split in minutes (default = 10)
        openai_api_key (str) : OpenAI API key. By default, the key is fetched from the `OPENAI_API_KEY` environment variable
    """
    time_start: float = time.time()
    print(f"Processing {file_name}...")
    audio_dir: str = split_file(file_name, split_size)

    if not only_split:
        transcription: str = get_transcript(
            audio_dir, openai_api_key, split_size, format)

        # Write result
        outfile: str = f"{os.path.basename(file_name).split(".")[0]}.{format}"
        print(f"Writing Transcript to {outfile}...")
        with open(outfile, "w", encoding="utf-8") as vtt_file:
            vtt_file.write(transcription)
    elapsed: float = time.time() - time_start
    print(f'Done in {int(elapsed)} seconds!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, epilog=textwrap.dedent(__doc__))
    parser.add_argument(
        "-f", "--file", help="Path to audio file to process", required=True)
    parser.add_argument(
        "-s", "--split_size", help="length of audio split in minutes (default=10)", default=10, type=int)
    parser.add_argument(
        "-o", "--output_format", help="Output format of the transcription (vtt or txt)", choices=["vtt", "txt"], default="vtt")
    parser.add_argument(
        "-k", "--openai_api_key", help="OpenAI API key", default="")
    parser.add_argument(
        "--only_split", help="Only split files without transcribing", action="store_true")
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s 1.0")

    args: argparse.Namespace = parser.parse_args()
    main(args.file, args.split_size, args.output_format,
         args.openai_api_key, args.only_split)
