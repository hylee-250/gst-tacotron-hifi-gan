from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import sys
sys.path.append('/workspace/tacotron2-gst/hifi_gan/')

import numpy as np
from scipy.io.wavfile import write

import torch

from sklearn.manifold import TSNE

from utils import load_wav_to_torch
from tqdm import tqdm
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from train import load_model
from text import text_to_sequence


import glob
import argparse
import json

from hifi_gan.env import AttrDict
from hifi_gan.meldataset import MAX_WAV_VALUE
from hifi_gan.models import Generator


hparams = create_hparams()

stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def load_mel(path):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec
    
checkpoint_path = "/workspace/tacotron2-gst/skt_multi_outdir/checkpoint_212500"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()


def TextEncoder(text):
    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    inputs = model.parse_input(sequence)
    embedded_inputs = model.embedding(inputs).transpose(1,2)
    transcript_outputs = model.encoder.inference(embedded_inputs)
    
    return transcript_outputs

def Decoder(encoder_outputs):
    decoder_input = model.decoder.get_go_frame(encoder_outputs)
    model.decoder.initialize_decoder_states(encoder_outputs, mask=None)
    mel_outputs, gate_outputs, alignments = [], [], []

    while True:
        decoder_input = model.decoder.prenet(decoder_input)
        mel_output, gate_output, alignment = model.decoder.decode(decoder_input)

        mel_outputs += [mel_output]
        gate_outputs += [gate_output]
        alignments += [alignment]

        if torch.sigmoid(gate_output.data) > hparams.gate_threshold:
            print(torch.sigmoid(gate_output.data), gate_output.data)
            break
        if len(mel_outputs) == hparams.max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output

    mel_outputs, gate_outputs, alignments = model.decoder.parse_decoder_outputs(
        mel_outputs, gate_outputs, alignments)
    mel_outputs_postnet = model.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
    return mel_outputs_postnet

def generate_mels_by_ref_audio(text, ref_audio):
    transcript_outputs = TextEncoder(text)
    print("ref_audio")
    ref_audio_mel = load_mel(ref_audio)
    latent_vector = model.gst(ref_audio_mel)
    latent_vector = latent_vector.expand_as(transcript_outputs)

    encoder_outputs = transcript_outputs + latent_vector
    
    mel_outputs = Decoder(encoder_outputs)
    return mel_outputs
    
def generate_mels_by_style_tokens(text):
    transcript_outputs = TextEncoder(text)
    GST = torch.tanh(model.gst.stl.embed)

    for idx in range(10):
        query = torch.zeros(1, 1, hparams.E//2).cuda()
        keys = GST[idx].unsqueeze(0).expand(1,-1,-1)
        style_emb = model.gst.stl.attention(query, keys)
        encoder_outputs = transcript_outputs + style_emb

        mel_outputs = Decoder(encoder_outputs)

        print("token {}".format(idx))
        return mel_outputs


#h = None
#device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    text = "이 모델을 이용하면 같은 문장을 여러가지 스타일로 말할 수 있습니다."
    mel_outputs = generate_mels_by_style_tokens(text)
    
    
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    #filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        # for i, filname in enumerate(filelist):
        #     x = np.load(os.path.join(a.input_mels_dir, filname))
        #     x = torch.FloatTensor(x).to(device)
        #     y_g_hat = generator(x)
        #     audio = y_g_hat.squeeze()
        #     audio = audio * MAX_WAV_VALUE
        #     audio = audio.cpu().numpy().astype('int16')

        #     output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
        #     write(output_file, h.sampling_rate, audio)
        #     print(output_file)
        x = mel_outputs
        #x = torch.FloatTensor(x)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        output_file = os.path.join(a.output_dir, 'skt_db' + '_generated_e2e.wav')
        write(output_file, h.sampling_rate, audio)
        print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], '/workspace/tacotron2-gst/hifi_gan/config_v1.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()