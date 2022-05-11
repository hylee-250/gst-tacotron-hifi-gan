###########################################################
#               GST-Tacotron Inference                    #                
###########################################################
import os
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import IPython.display as ipd

import sys
sys.path.append('/workspace/tacotron2-gst/waveglow/')

import numpy as np
from scipy.io.wavfile import write

#import glow
import torch

from sklearn.manifold import TSNE

from utils import load_wav_to_torch
from tqdm import tqdm
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from train import load_model
from text import text_to_sequence



hparams = create_hparams()
hparams.sampling_rate = 16000
hparams.max_decoder_steps = 1000

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

def plot_data(data, figsize=(16, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(data, aspect='auto', origin='bottom', interpolation='none')
    
checkpoint_path = "/workspace/tacotron2-gst/skt_outdir2/checkpoint_103000"
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

    with torch.no_grad():
        synth = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        
    return synth, mel_outputs_postnet

def generate_mels_by_ref_audio(text, ref_audio):
    transcript_outputs = TextEncoder(text)
    print("ref_audio")
    ipd.display(ipd.Audio(ref_audio, rate=hparams.sampling_rate))
    ref_audio_mel = load_mel(ref_audio)
    ipd.display(plot_data(ref_audio_mel.data.cpu().numpy()[0]))
    latent_vector = model.gst(ref_audio_mel)
    latent_vector = latent_vector.expand_as(transcript_outputs)

    encoder_outputs = transcript_outputs + latent_vector
    
    synth, mel_outputs = Decoder(encoder_outputs)
    
    ipd.display(ipd.Audio(synth[0].data.cpu().numpy(), rate=hparams.sampling_rate))
    ipd.display(plot_data(mel_outputs.data.cpu().numpy()[0]))
    
#######################################
#   Reference Audio version
#######################################    
# text = "이 모델을 이용하면 같은 문장을 여러가지 스타일로 말할 수 있습니다."
# ref_wav = "/workspace/data/Emo_kor/acriil_ang_00000100.wav"
# generate_mels_by_ref_audio(text, ref_wav)


def generate_mels_by_style_tokens(text):
    transcript_outputs = TextEncoder(text)
    GST = torch.tanh(model.gst.stl.embed)

    for idx in range(10):
        query = torch.zeros(1, 1, hparams.E//2).cuda()
        keys = GST[idx].unsqueeze(0).expand(1,-1,-1)
        style_emb = model.gst.stl.attention(query, keys)
        encoder_outputs = transcript_outputs + style_emb

        synth, mel_outputs = Decoder(encoder_outputs)

        print("token {}".format(idx))
        ipd.display(ipd.Audio(synth[0].data.cpu().numpy(), rate=hparams.sampling_rate))
        ipd.display(plot_data(mel_outputs.data.cpu().numpy()[0]))

#######################################
#   Style Tokens version
####################################### 
# text = "이 모델을 이용하면 같은 문장을 여러가지 스타일로 말할 수 있습니다."
# generate_mels_by_style_tokens(text)





##########################################################

##########################################################
#         TransformerTTS + Hifi-GAN synthesis            #
##########################################################
from preprocess import get_dataset, DataLoader, collate_fn_transformer
import torch
from utils import spectrogram2wav, update_kv_mask
from scipy.io.wavfile import write

#import hyperparams as hp
import hparams as hp

#from text.HangulUtilsHrim import hangul_to_sequence
#from text import text_to_sequence


import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
from util.writer import get_writer
import json

import librosa
import matplotlib
matplotlib.use('Agg')
import librosa.display
import matplotlib.pyplot as plt
import time

import os
import sys

#from mel2audio.args import parse_args
#from mel2audio.hps import Hyperparameters
#from mel2audio.model import SmartVocoder

#import soundfile

from hifi_gan.models import Generator
from hifi_gan.env import AttrDict

def load_checkpoint(step, model_name="transformer"):
    state_dict = torch.load('./checkpoints/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def resample(x, scale, seq_len):
    device = x.device
    batch_size = x.size(0)
    indices = torch.arange(2*seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    idx_scaled = indices / scale	# scales must be 0.5~2
    idx_scaled_fl = torch.floor(idx_scaled)
    lambda_ = idx_scaled - idx_scaled_fl

    target_mask = idx_scaled_fl < (seq_len-1)
    target_len = target_mask.sum(dim=-1)

    index_1 = torch.repeat_interleave(torch.arange(batch_size, device=device), target_len)
    idx_2_fl = idx_scaled_fl[target_mask].long()
    idx_2_cl = idx_2_fl + 1
    y_fl = x[index_1, idx_2_fl, :]
    y_cl = x[index_1, idx_2_cl, :]

    lambda_f = lambda_[target_mask].unsqueeze(-1)
    y = (1-lambda_f)*y_fl + lambda_f*y_cl
    return y.unsqueeze(0).repeat(batch_size, 1, 1)
	
def synthesis(args):
    checkpoint_path = "/workspace/tacotron2-gst/skt_outdir2/checkpoint_103000"
    model = load_model(hp)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.eval()
    
    #m = Model()
    #m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))   
    #m=m.cuda()
    #m.train(False)
    
    
#    vocoder = SmartVocoder(Hyperparameters(parse_args()))
#    vocoder.load_state_dict(t.load('./mel2audio/merged_STFT_checkpoint.pth')["state_dict"])
#    vocoder=vocoder.cuda()
#    vocoder.eval()
    with open('./hifi_gan/config.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    hifi_gan = Generator(h).cuda()
    state_dict_g = torch.load('./hifi_gan/g_00334000', map_location='cuda')
    hifi_gan.load_state_dict(state_dict_g['generator'])
    hifi_gan.eval()
    hifi_gan.remove_weight_norm()


    test_dataset = get_dataset(hp.test_data_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
    ref_dataset = get_dataset(hp.test_data_csv_shuf)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)

    writer = get_writer(hp.checkpoint_path, hp.log_directory)

    mel_basis = torch.from_numpy(librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels, 50, 11000)).unsqueeze(0)  # (n_mels, 1+n_fft//2)
    
    ref_dataloader_iter = iter(ref_dataloader)
    _, ref_mel, _, _, _, ref_pos_mel, _, _, ref_fname = next(ref_dataloader_iter)
   
    for i, data in enumerate(test_dataloader):
        character, _, _, _, pos_text, _, text_length, _, fname = data
        mel_input = torch.zeros([1,1,80]).cuda()
        character = character.cuda()
        ref_mel = ref_mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        with torch.no_grad():
            start=time.time()
            memory, c_mask, attns_enc, duration_mask = m.encoder(character, pos=pos_text)
            style, coarse_emb = model.ref_encoder(ref_mel)
            memory = torch.cat((memory, coarse_emb.expand(-1, memory.size(1), -1)), -1)
            memory = model.memory_coarse_layer(memory)
            duration_predictor_output = model.duration_predictor(memory, duration_mask)
            duration = torch.ceil(duration_predictor_output)
            duration = duration * duration_mask
#            max_length = t.sum(duration).type(t.LongTensor)
#            print("length : ", max_length)

            monotonic_interpolation, pos_mel_, weights = m.length_regulator(memory, duration, duration_mask)
            kv_mask = torch.zeros([1, mel_input.size(1), character.size(1)]).cuda()		# B, t', N
            kv_mask[:, :, :3] = 1
            kv_mask = kv_mask.eq(0)
            stop_flag = False
            ctr = 0
            for j in range(1200):
                pos_mel = torch.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
                mel_pred, postnet_pred, attn_probs, decoder_output, attns_dec, attns_style = m.decoder(memory, style, mel_input, c_mask,
                                                                                                     pos=pos_mel, ref_pos=ref_pos_mel, mono_inter=monotonic_interpolation[:,:mel_input.shape[1]], kv_mask=kv_mask)
                mel_input = torch.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)
#                print("j", j, "mel_input", mel_input.shape)
                if stop_flag and ctr == 10:
                    break
                elif stop_flag:
                    ctr += 1
                kv_mask, stop_flag = update_kv_mask(kv_mask, attn_probs)		# B, t', N --> B, t'+1, N
            postnet_pred = torch.cat((postnet_pred, t.zeros(postnet_pred.size(0), 5, postnet_pred.size(-1)).cuda()), 1)
            gen_length = mel_input.size(1)
#            print("gen_length", gen_length)
            post_linear = model.postnet(postnet_pred)
            post_linear = resample(post_linear, seq_len=mel_input.size(1), scale=args.rhythm_scale)
            postnet_pred = resample(mel_input, seq_len=mel_input.size(1), scale=args.rhythm_scale)
            inf_time = time.time() - start
            print("inference time: ", inf_time)
#            print("speech_rate: ", len(postnet_pred[0])/len(character[0]))

            postnet_pred_v = postnet_pred.transpose(2,1)
            postnet_pred_v = (postnet_pred_v*100+20-100)/20
            B, C, T = postnet_pred_v.shape
            z = torch.randn(1, 1, T*hp.hop_length).cuda()
            z = z * 0.6 	# Temp
#            t.cuda.synchronize()
#            timestemp = time.time()
#            with t.no_grad():
#                y_gen = vocoder.reverse(z, postnet_pred_v).squeeze()
#            t.cuda.synchronize()
#            print('{} seconds'.format(time.time() - timestemp))
#            wav = y_gen.to(t.device("cpu")).data.numpy()
#            wav = np.pad(wav, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end

            post_linear_v = post_linear.transpose(1,2)
            post_linear_v = 10**((post_linear_v*100+20-100)/20)
            mel_basis = mel_basis.repeat(post_linear_v.shape[0], 1, 1)
            post_linear_mel_v = torch.log10(torch.bmm(mel_basis.cuda(),post_linear_v))
            B, C, T = post_linear_mel_v.shape
            z = torch.randn(1, 1, T*hp.hop_length).cuda()
            z = z * 0.6 	# Temp
#            t.cuda.synchronize()
#            timestemp = time.time()
#            with t.no_grad():
#                y_gen_linear = vocoder.reverse(z, post_linear_mel_v).squeeze()
#            t.cuda.synchronize()
#            wav_linear = y_gen_linear.to(t.device("cpu")).data.numpy()
#            wav_linear = np.pad(wav_linear, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end

            wav_hifi = hifi_gan(post_linear_mel_v).squeeze().clamp(-1,1).detach().cpu().numpy()
            wav_hifi = np.pad(wav_hifi, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end


        mel_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'mel')
        if not os.path.exists(mel_path):
            os.makedirs(mel_path)
        np.save(os.path.join(mel_path, 'text_{}_ref_{}_synth_{}.mel'.format(i, ref_fname, str(args.rhythm_scale))), postnet_pred.cpu())       

        linear_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'linear')
        if not os.path.exists(linear_path):
            os.makedirs(linear_path)
        np.save(os.path.join(linear_path, 'text_{}_ref_{}_synth_{}.linear'.format(i, ref_fname, str(args.rhythm_scale))), post_linear.cpu())       

#        wav_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav')
#        if not os.path.exists(wav_path):
#            os.makedirs(wav_path)
#        write(os.path.join(wav_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav)
#        print("rtx : ", (len(wav)/hp.sr) / inf_time)

#        wav_linear_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav_linear')
#        if not os.path.exists(wav_linear_path):
#            os.makedirs(wav_linear_path)
#        write(os.path.join(wav_linear_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav_linear)

        wav_hifi_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav_hifi')
        if not os.path.exists(wav_hifi_path):
            os.makedirs(wav_hifi_path)
        write(os.path.join(wav_hifi_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav_hifi)  

        show_weights = weights.contiguous().view(weights.size(0), 1, 1, weights.size(1), weights.size(2))
        attns_enc_new=[]
        attns_dec_new=[]
        attn_probs_new=[]
        attns_style_new=[]
        for i in range(len(attns_enc)):
            attns_enc_new.append(attns_enc[i].unsqueeze(0))
            attns_dec_new.append(attns_dec[i].unsqueeze(0))
            attn_probs_new.append(attn_probs[i].unsqueeze(0))
            attns_style_new.append(attns_style[i].unsqueeze(0))
        attns_enc = torch.cat(attns_enc_new, 0)
        attns_dec = torch.cat(attns_dec_new, 0)
        attn_probs = torch.cat(attn_probs_new, 0)
        attns_style = torch.cat(attns_style_new, 0)

        attns_enc = attns_enc.contiguous().view(attns_enc.size(0), 1, hp.n_heads, attns_enc.size(2), attns_enc.size(3))
        attns_enc = attns_enc.permute(1,0,2,3,4)
        attns_dec = attns_dec.contiguous().view(attns_dec.size(0), 1, hp.n_heads, attns_dec.size(2), attns_dec.size(3))
        attns_dec = attns_dec.permute(1,0,2,3,4)
        attn_probs = attn_probs.contiguous().view(attn_probs.size(0), 1, hp.n_heads, attn_probs.size(2), attn_probs.size(3))
        attn_probs = attn_probs.permute(1,0,2,3,4)
        attns_style = attns_style.contiguous().view(attns_style.size(0), 1, hp.n_heads, attns_style.size(2), attns_style.size(3))
        attns_style = attns_style.permute(1,0,2,3,4)

        save_dir = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'figure', "text_{}_ref_{}_synth_{}.wav".format(fname, ref_fname, str(args.rhythm_scale)))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        writer.add_alignments(attns_enc.detach().cpu(), attns_dec.detach().cpu(), attn_probs.detach().cpu(), attns_style.detach().cpu(), show_weights.detach().cpu(), [torch.tensor(gen_length).type(torch.LongTensor)] ,text_length, args.restore_step1, 'Inference', save_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=694000)
    parser.add_argument('--rhythm_scale', type=float, help='Global step to restore checkpoint', default=1.)


    args = parser.parse_args()
    synthesis(args)

########################################################################
#                   Hifi-GAN inference
########################################################################
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='/workspace/data/skt_db/2020/small/FBSE0/wav_48000')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config_v1.json')
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



########################################################################
#                   Hifi-GAN inference_e2e
########################################################################
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

h = None
device = None


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
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
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





#######################################################################
#           GST-Tacotron + Hifi-GAN Synthesis
#######################################################################
import os
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import sys

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

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import argparse
import json

from hifi_gan.env import AttrDict
from hifi_gan.meldataset import MAX_WAV_VALUE
from hifi_gan.models import Generator


hparams = create_hparams()
hparams.sampling_rate = 48000
hparams.max_decoder_steps = 1000

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
    
checkpoint_path = "/workspace/tacotron2-gst/skt_outdir2/checkpoint_103000"
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


h = None
device = None


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

    filelist = os.listdir(a.input_mels_dir)

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
        x = torch.FloatTensor(x).to(device)
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

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
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


























from preprocess import get_dataset, DataLoader, collate_fn_transformer
from utils import spectrogram2wav, update_kv_mask
from scipy.io.wavfile import write


from network import ModelPostNet, Model
from collections import OrderedDict
import argparse
from util.writer import get_writer
import json

import librosa
import matplotlib
matplotlib.use('Agg')
import librosa.display
import matplotlib.pyplot as plt
import time

from hifi_gan.models import Generator
from hifi_gan.env import AttrDict

def synthesis(args):
    checkpoint_path = "/workspace/tacotron2-gst/skt_outdir2/checkpoint_103000"
    model = load_model(hp)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.eval()

    with open('./hifi_gan/config.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    hifi_gan = Generator(h).cuda()
    state_dict_g = torch.load('./hifi_gan/g_00334000', map_location='cuda')
    hifi_gan.load_state_dict(state_dict_g['generator'])
    hifi_gan.eval()
    hifi_gan.remove_weight_norm()