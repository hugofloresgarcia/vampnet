import time
from pathlib import Path
import shutil
import json

import argbind
import audiotools as at
from gradio_client import Client, handle_file
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
import torch

class Timer:
    
    def __init__(self):
        self.times = {}
    
    def tick(self, name: str):
        self.times[name] = time.time()
    
    def tock(self, name: str):
        toc = time.time() - self.times[name]
        print(f"{name} took {toc} seconds")
        return toc
    
    def __str__(self):
        return str(self.times)

timer = Timer()

DOWNLOADS_DIR = ".gradio"

def clear_file(file):
    file = Path(file)
    if file.exists():
        file.unlink()


class OSCManager:

    def __init__(
        self, 
        ip: str, 
        s_port: str, 
        r_port: str,
        process_fn: callable, 
        # param_change_callback: callable = None
    ):
        self.ip = ip
        self.s_port = s_port
        self.r_port = r_port

        # register the process_fn
        self.process_fn = process_fn

        print(f"will send to {ip}:{s_port}")
        self.client = SimpleUDPClient(ip, s_port)


    def start_server(self,):
        dispatcher = Dispatcher()
        dispatcher.map("/process", self.process_fn)

        def send_heartbeat(_, *args):
            # print("Received heartbeat")
            self.client.send_message("/heartbeat", "pong")

        dispatcher.map("/heartbeat", lambda a, *r: send_heartbeat(a, *r))

        dispatcher.map("/cleanup", lambda a, *r: clear_file(r[0]))

        dispatcher.set_default_handler(lambda a, *r: print(a, r))

        server = ThreadingOSCUDPServer((self.ip, self.r_port), dispatcher)
        print(f"Serving on {server.server_address}")
        server.serve_forever()

    def error(self, msg: str):
        self.client.send_message("/error", msg)

    def log(self, msg: str):
        self.client.send_message("/log", msg)
        

class GradioOSCClient:

    def __init__(self, 
        ip: str,
        s_port: int, r_port: int,
        vampnet_url: str = None, # url for vampnet
        s2s_urls: str = None, # urls for sketch2sound (multiple for parallel processing)
        s2s_vim_url: str = None, # for sketch2sound old version (good at vocal imitations, (vims))
    ):
        self.osc_manager = OSCManager(
            ip=ip, s_port=s_port, r_port=r_port, 
            process_fn=self.process, 
        )

        self.clients = {}
        if vampnet_url is not None:
            self.clients["vampnet"] = Client(src=vampnet_url, download_files=DOWNLOADS_DIR)
        if s2s_urls is not None:
            # self.clients["s2s"] = Client(src=s2s_url, download_files=DOWNLOADS_DIR)
            for i, s2s_url in enumerate(s2s_urls):
                if s2s_url is not None:
                    self.clients[f"s2s_{i}"] = Client(src=s2s_url, download_files=DOWNLOADS_DIR)
        if s2s_vim_url is not None:
            self.clients["s2s_vim"] = Client(src=s2s_vim_url, download_files=DOWNLOADS_DIR)
        
        assert len(self.clients) > 0, "At least one client must be specified!"

        self.batch_size = 2# TODO: automatically get batch size from client. 

        self.osc_manager.log("hello from gradio client!")

        self.inf_idx = 0


    def param_changed(self, param_name, new_value):
        print(f"Parameter {param_name} changed to {new_value}")

    def vampnet_process(self, address: str, *args):
        client = self.clients["vampnet"]

        # query id --- audiofile ---- model_choice --- periodic --- drop --- seed 
        query_id = args[0]
        client_type = args[1]
        audio_path = Path(args[2])
        model_choice = args[3]
        periodic_p = args[4]
        dropout = args[5]
        seed = args[6]
        looplength_ms = args[7]
        typical_filter = args[8]
        typical_mass = args[9]
        typical_min_tokens = args[10]
        upper_codebook_mask = args[11]
        onset_mask_width = args[12]
        sampling_steps = args[13]
        temperature = args[14]
        top_p = args[15]
        beat_mask_ms = args[16]
        num_feedback_steps  = args[17]
        
        if not audio_path.exists():
            print(f"File {audio_path} does not exist")
            self.osc_manager.error(f"File {audio_path} does not exist")
            return

        sig = at.AudioSignal(audio_path)
        sig.to_mono()
        sig.sample_rate = 48000 # HOT PATCH (FIXME IN MAX: sample rate is being forced to 48k)

        # grab the looplength only
        # TODO: although I added this, 
        # the max patch is still configured to crop anything past the looplength off
        # so we'll have to change that in order to make an effect. 
        end_sample = int((looplength_ms * sig.sample_rate) / 1000)

        # grab  the remainder of the waveform
        num_cut_samples = sig.samples.shape[-1] - end_sample
        cut_wav = sig.samples[..., -num_cut_samples:]

        sig.samples = sig.samples[..., :end_sample]
        # write the file back
        sig.write(audio_path)
        
        timer.tick("predict")
        print(f"Processing {address} with args {args}")
        # breakpoint()
        job = client.submit(
            input_audio=handle_file(audio_path),
            sampletemp=temperature,
            top_p=top_p,
            periodic_p=periodic_p,
            periodic_w=1,
            dropout=dropout,
            stretch_factor=1,
            onset_mask_width=onset_mask_width,
            typical_filtering=bool(typical_filter),
            typical_mass=typical_mass,
            typical_min_tokens=typical_min_tokens,
            seed=seed,
            model_choice=model_choice,
            n_mask_codebooks=upper_codebook_mask,
            pitch_shift_amt=0,
            sample_cutoff=1.0,
            sampling_steps=sampling_steps,
            beat_mask_ms=int(beat_mask_ms),
            num_feedback_steps=num_feedback_steps,
            api_name="/vamp_1"
        )

        while not job.done():
            time.sleep(0.1)
            self.osc_manager.client.send_message("/progress", [query_id, str(job.status().code)])

        result = job.result()
        # audio_file = result
        # audio_files = [audio_file] * self.batch_size
        audio_files = list(result[:self.batch_size])
        # if each file is missing a .wav at the end, add it 
        first_audio = audio_files[0]
        if not first_audio.endswith(".wav"):
            for audio_file in set(audio_files):
                if not audio_file.endswith(".wav"):
                    shutil.move(audio_file, f"{audio_file}.wav")
                    audio_file = f"{audio_file}.wav"
            audio_files = [f"{audio}.wav" for audio in audio_files if not audio.endswith(".wav")]
        
        for audio_file in audio_files:
            # load the file, add the cut samples back
            sig = at.AudioSignal(audio_file)
            sig.resample(48000)
            sig.samples = torch.cat([sig.samples, cut_wav], dim=-1)
            sig.write(audio_file)
        seed = result[-1]

        timer.tock("predict")

        # send a message that the process is done
        self.osc_manager.log(f"query {query_id} has been processed")
        self.osc_manager.client.send_message("/process-result", [query_id] + audio_files)

   
    def process(self, address: str, *args):
        query_id = args[0]
        client_type = args[1]
        audio_path = Path(args[2])

        if client_type == "vampnet":
            self.vampnet_process(address, *args)
            return
        elif client_type == "sketch2sound":
            self.process_s2s(address, *args)
            return
        else:
            raise ValueError(f"Unknown client type {client_type}")
        
    def process_s2s(self, address: str, *args):
        # client = self.clients["s2s"]

        if address != "/process":
            raise ValueError(f"Unknown address {address}")

        print(f"Processing {address} with args {args}")
        # unpack the args
        query_id = args[0]
        client_type = args[1]
        audio_path = Path(args[2])
        text_prompt = args[3]
        use_control = args[4] == 1
        looplength = args[5]
        guidance_scale = args[6]
        seed = args[7]
        median_filter_length = args[8]

        sig = at.AudioSignal(audio_path)
        looplength_ms = looplength
        # grab the looplength only
        # TODO: although I added this, 
        # the max patch is still configured to crop anything past the looplength off
        # so we'll have to change that in order to make an effect. 
        end_sample = int(looplength_ms * sig.sample_rate / 1000)
        sig.samples = sig.samples[..., :end_sample]

        # grab  the remainder of the waveform
        num_cut_samples = sig.samples.shape[-1] - end_sample
        cut_wav = sig.samples[..., -num_cut_samples:]

        # write the file back
        sig.write(audio_path)

        # make sure it exists, otherwise send an error message
        if not audio_path.exists():
            print(f"File {audio_path} does not exist")
            self.osc_manager.error(f"File {audio_path} does not exist")
            return

        
        timer.tick(f"predict-{query_id}")
        # NEW API
        if use_control and "s2s_vim" in self.clients:
            client = self.clients["s2s_vim"]
            job = client.submit(
                data=handle_file(audio_path),
                param_1=-20, # comp threshold
                param_2=1, # comp ratio
                param_3=seed, # random seed
                param_4=text_prompt, # text prompt
                param_5=5, # text guid
                param_6=1, # ctrl guid
                param_7=-1,  # t_low
                param_8=1, # t_high
                param_9=15, # median filt
                param_10=True, # use centroid
                param_11=True, # use pitch
                param_12=True, # use ldns
                param_13=0, # transpose (semis)
                param_14=0, # pitch shift semis
                param_15=0, # pitchiness ofset
                param_16=0, # gain db
                param_17=100, # num steps
                api_name="/generate"
            )
        else: 
            client = self.clients[f"s2s_{self.inf_idx % len(self.clients)}"]
            self.inf_idx += 1
            params = {  
                    'control_guidance_scale': 1.0,
                    'guidance_scale': guidance_scale,
                    'logsnr_max': 5.0,
                    'logsnr_min': -8,
                    'num_seconds': looplength / 1000.,
                    'num_steps': 24,
                    'rho': 7.0,
                    'sampler': 'dpmpp-2m-sde',
                    'schedule': 'karras'
            }
            job = client.submit(
                    text_prompt=text_prompt,
                    control_audio=handle_file(audio_path) if use_control else None,
                    seed=seed,
                    median_filter_length=median_filter_length,
                    normalize_db=-16,
                    duration=looplength / 1000.,
                    params_str=json.dumps(params),
                    api_name="/generate_with_params"
            )

        while not job.done():
            time.sleep(0.1)
            status = str(job.status().code)

            if "FINISHED" in status:
                status = "STATUS.WAITING"
        
            self.osc_manager.client.send_message("/progress", [query_id, status])

        result = job.result()
        self.osc_manager.client.send_message("/progress", [query_id, "STATUS.FINISHED"])
        timer.tock(f"predict-{query_id}")
        timer.tick(f"postprocess-{query_id}")
        audio_files = list(result[:self.batch_size])
        # if each file is missing a .wav at the end, add it 
        first_audio = audio_files[0]
        if not first_audio.endswith(".wav"):
            for audio_file in set(audio_files):
                if not audio_file.endswith(".wav"):
                    shutil.move(audio_file, f"{audio_file}.wav")
            audio_files = [f"{audio}.wav" for audio in audio_files]
        
        for audio_file in audio_files:
            # load the file, add the cut samples back
            sig = at.AudioSignal(audio_file)
            sig = sig.to_mono()
            sig.samples = torch.cat([sig.samples, cut_wav], dim=-1)
            sig.write(audio_file)
        seed = result[-1]
        timer.tock(f"postprocess-{query_id}")

        # send a message that the process is done
        self.osc_manager.log(f"query {query_id} has been processed")
        self.osc_manager.client.send_message("/process-result", [query_id] + audio_files)

        # schedule to delete the file after 30 seconds
        time.sleep(30)
        print("deleting files from query", query_id)
        for audio_file in audio_files:
            clear_file(audio_file)
        # import shutil
        print(str(Path(audio_files[0]).parent))
        shutil.rmtree(str(Path(audio_files[0]).parent))


def gradio_main(
    s2s_url: str = None, 
    s2s_url_2: str = None,
    s2s_vim_url: str = None,
    vampnet_url: str = None
):
    s2s_urls = [s2s_url, s2s_url_2]
    system = GradioOSCClient(
        vampnet_url=vampnet_url,
        s2s_urls=s2s_urls,
        s2s_vim_url=s2s_vim_url,
        ip="127.0.0.1", s_port=8003, r_port=8001,
    )

    system.osc_manager.start_server()


if __name__ == "__main__":
    try:
        gradio_main = argbind.bind(gradio_main, without_prefix=True)

        args = argbind.parse_args()
        with argbind.scope(args):
            gradio_main()

    except Exception as e:
        import shutil
        shutil.rmtree(DOWNLOADS_DIR, ignore_errors=True)
        raise e