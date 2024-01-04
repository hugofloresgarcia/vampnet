#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encodes or decodes an audio file with the descript audio codec.
Compared to the included "python3 -m dac encode" and "python3 -m dac decode"
of descript-audio-codec 1.0.0, this implementation differs in the following:
- chunked encoding produces the same codes as unchunked encoding,
  except at the beginning and end of the file
- encoded files are in .npz format, so the codes can be accessed as a memory
  map and do not need to be unpickled
- the codes are stored in C order with the time dimension first and the
  channels last, so accessing a temporal excerpt is efficient
- by default, decoding does not restore the original sample rate, but supports
  so via --resample

For usage information, call with --help.

Author: Jan Schlüter
"""

from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import torch
import tqdm
from dac.utils import load_model
from audiotools import AudioSignal


def opts_parser():
    usage =\
"""Encodes or decodes an audio file with the descript audio codec.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        'command',
        type=str, choices=('encode', 'decode', 'test'),
        help='The action to perform.')
    parser.add_argument(
        'infile',
        type=str,
        help='A sound file for encoding, or dac file for decoding, or a '
        'directory of such files.')
    parser.add_argument(
        'outfile',
        type=str,
        help='A dac file for encoding, or a sound file for decoding, or a '
        'directory of such files.')
    parser.add_argument(
        '-d', '--device',
        type=str, default='auto',
        help='Device to use: cpu, cuda[:n], auto (default: %(default)s)')
    parser.add_argument(
        '-w', '--win_duration',
        type=float, default=5,
        help='Chunk duration in seconds (default: %(default)s)')
    parser.add_argument(
        '-m', '--model', 
        type=str, default='./models/codec.pth',
        help='Path to the model file (default: %(default)s)')
    parser.add_argument(
        '-r', '--resample',
        action='store_true',
        help='If given, restores original sample rate and size on decoding.')
    return parser


def load_audio(infile, sample_rate=None, normalize_db=-16):
    """
    Reads the given audio file into an AudioSignal, optionally resampled
    and normalized. Returns the audio signal and a metadata dictionary
    containing its original sample count ('original_length'), its original
    sample rate ('sample_rate'), its original loudness ('input_db').
    If normalize_db is None, then 'input_db' is omitted.
    """
    audio = AudioSignal(infile)
    metadata = dict(original_length=audio.shape[-1],
                    sample_rate=audio.sample_rate)
    if sample_rate is not None:
        audio.resample(sample_rate)
    if normalize_db is not None:
        metadata['input_db'] = audio.loudness().cpu().numpy()
        audio.normalize(normalize_db)
    audio.ensure_max_of_audio()
    batchsize, channels, time = audio.shape
    audio.audio_data = audio.audio_data.view(batchsize * channels, 1, time)
    return audio, metadata


def receptive_field(model):
    """
    Computes the size, stride and padding of the given model's receptive
    field under the assumption that all its Conv1d and TransposeConv1d
    layers are applied in sequence.
    """
    total_size, total_stride, total_padding = 1, 1, 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            layer_size = layer.dilation[0] * (layer.kernel_size[0] - 1) + 1
        if isinstance(layer, torch.nn.Conv1d):
            # update size
            total_size += (layer_size - 1) * total_stride
            # update padding
            total_padding += layer.padding[0] * total_stride
            # update stride
            total_stride *= layer.stride[0]
        elif isinstance(layer, torch.nn.ConvTranspose1d):
            # update stride
            total_stride /= layer.stride[0]
            # update padding
            total_padding += (layer_size - layer.padding[0]) * total_stride
            # update size
            total_size += (layer_size - 1) * total_stride
    return total_size, total_stride, total_padding


@torch.inference_mode()
def compress(model, device, audio, win_duration, n_quantizers=None):
    """Encodes the given audio signal, returns the codes."""
    # right-pad to the next multiple of hop length
    # (as the model's internal padding is short by one hop length)
    remainder = audio.shape[-1] % model.hop_length
    right_pad = model.hop_length - remainder if remainder else 0
    if not win_duration:
        model.padding = True
        if right_pad:
            audio.zero_pad(0, right_pad)
        samples = audio.audio_data.to(device)
        codes = model.encode(samples, n_quantizers)[1]
        codes = codes.permute(2, 1, 0).short()  # -> time, quantizers, channels
    else:
        # determine receptive field of encoder
        model.padding = True
        field_size, stride, padding = receptive_field(model.encoder)
        model.padding = False
        # determine the window size to use
        # - the maximum samples the user wants to read at once
        win_size = int(win_duration * model.sample_rate)
        # - how many code frames we would get from this
        num_codes = (win_size - field_size + stride) // stride
        # - how many samples are actually involved in that
        win_size = field_size + (num_codes - 1) * stride
        # determine the hop size to use
        hop_size = num_codes * stride
        # finally process the input
        codes = []
        audio_size = audio.audio_data.size(-1)
        for start_position in tqdm.trange(-padding,
                                          audio_size + padding + right_pad,
                                          hop_size,
                                          leave=False):
            # extract chunk
            chunk = audio[..., max(0, start_position):start_position + win_size]
            # zero-pad the first chunk(s)
            if start_position < 0:
                chunk.zero_pad(-start_position, 0)
            chunk_size = chunk.audio_data.size(-1)
            # skip the last chunk if it would not have yielded any output
            if chunk_size + padding + right_pad < field_size:
                continue
            # pad the last chunk(s) to the full window size if needed
            if chunk_size < win_size:
                chunk.zero_pad(0, win_size - chunk_size)
            # process chunk
            samples = chunk.audio_data.to(device)
            c = model.encode(samples, n_quantizers)[1].cpu()
            c = c.permute(2, 1, 0)  # -> time, quantizers, channels
            # remove excess frames from padding if needed
            if chunk_size + padding + right_pad < win_size:
                chunk_codes = (chunk_size + padding + right_pad - field_size + stride) // stride
                c = c[:chunk_codes]
            codes.append(c.short())
        codes = torch.cat(codes, dim=0)
    return codes.contiguous()


def save_dac(outfile, codes, **metadata):
    """
    Writes the given codes to the given output file, with optional metadata.
    """
    try:
        with open(outfile, 'wb') as f:  # to allow a custom file extension
            np.savez(f, codes=codes, metadata=np.asarray(metadata, dtype='O'))
    except KeyboardInterrupt:
        Path(outfile).unlink()  # avoid half-written files
        raise

def encode(model, device, infile, outfile, win_duration, n_quantizers=None,
           normalize_db=-16):
    """Encodes the given audio file, writes codes to the given output file."""
    audio, metadata = load_audio(infile, model.sample_rate, normalize_db)
    codes = compress(model, device, audio, win_duration, n_quantizers)
    save_dac(outfile, codes, **metadata)


def load_dac(infile):
    """Reads codes and metadata from the given DAC file."""
    with np.load(infile, allow_pickle=True) as f:
        return torch.as_tensor(f['codes']), f['metadata'].item()


@torch.inference_mode()
def decompress(model, device, codes, win_duration):
    """Decodes the given codes, returns the audio signal."""
    if not win_duration:
        model.padding = True
        codes = codes.permute(2, 1, 0).to(device).int()
        latents = model.quantizer.from_codes(codes)[0]
        samples = model.decode(latents)
    else:
        raise NotImplementedError("Chunked decoding not implemented")
    audio = AudioSignal(samples, sample_rate=model.sample_rate)
    return audio


def save_audio(outfile, audio):
    """Writes the given audio signal to the given output file."""
    channels, _, time = audio.shape
    audio.audio_data = audio.audio_data.view(1, channels, time)
    audio.write(outfile)


def decode(model, device, infile, outfile, win_duration, resample=False):
    """Decodes the given DAC file, writes audio to the given output file."""
    codes, metadata = load_dac(infile)
    audio = decompress(model, device, codes, win_duration)
    if metadata.get('input_db', None) is not None:
        audio.normalize(metadata['input_db'])
    if resample:
        audio.resample(metadata['sample_rate'])
    if audio.sample_rate == metadata['sample_rate']:
        audio = audio[..., :metadata['original_length']]
    save_audio(outfile, audio)


def recursively(func, out_ext, model, device, indir, outdir, *args):
    """
    Apply encode() or decode() over a directory, using out_ext as the
    output file extension.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for inpath in tqdm.tqdm(list(indir.rglob("*"))):
        outpath = outdir / inpath.relative_to(indir)
        if inpath.is_dir():
            outpath.mkdir(parents=True, exist_ok=True)
        else:
            outpath = outpath.with_suffix(out_ext)
            try:
                outpath.touch(exist_ok=False)
            except FileExistsError:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    func(model, device, inpath, outpath, *args)
            except Exception as e:
                print("Skipping %r: %r" % (inpath, e))


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # pick device
    if options.device == 'auto':
        options.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(options.device)

    # load model
    model = load_model(load_path=options.model)
    model.to(device)
    model.eval()
    print("Receptive fields (size, stride, padding):")
    print("encoder:", receptive_field(model.encoder))
    print("decoder:", receptive_field(model.decoder))
    print("total:", receptive_field(model))

    # distinguish actions
    if options.command in ('encode', 'decode'):
        infile = Path(options.infile)
        outfile = Path(options.outfile)
        args = (model, device, infile, outfile, options.win_duration)
    if options.command == 'encode':
        if infile.is_dir():
            recursively(encode, '.dac', *args)
        else:
            encode(*args)
    elif options.command == 'decode':
        args = args + (options.resample,)
        if infile.is_dir():
            recursively(decode, '.wav', *args)
        else:
            decode(*args)
    elif options.command == 'test':
        audio, _ = load_audio(options.infile)
        print("computing unchunked encoding...")
        codes1 = compress(model, device, audio, win_duration=0)
        print("computing chunked encoding...")
        codes2 = compress(model, device, audio, win_duration=1.0)
        print("encoding is agnostic:", np.allclose(codes1, codes2))
        print("encoding is agnostic ignoring the first/last 6 frames:",
              np.allclose(codes1[6:-6], codes2[6:-6]))


if __name__ == "__main__":
    main()