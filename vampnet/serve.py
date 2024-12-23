from threading import Lock
import time
from dataclasses import dataclass
from pathlib import Path

from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

import torch
import vampnet
import vampnet.signal as sn
from vampnet.mask import apply_mask
from tqdm import tqdm

ip = "127.0.0.1"
s_port = 1335
r_port = 1334

@dataclass
class Param:
    name: str
    value: any
    _t: type




def set_filter(address: str, *args: list[any]) -> None:
    # We expect two float arguments
    if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
        return

    # Check that address starts with filter
    if not address[:-1] == "/filter":  # Cut off the last character
        return

# a timer that allows recording things with timer.tick(name="name")
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

# a class that wraps the vampnet interface
# and provides a way to interact with it
# over OSC with digital musical instruments
class VampNetDigitalInstrumentSystem:
    CORRUPT_TYPES = ("periodic", "random", "onsets", "all")
    SAMPLING_STEPS = 12
    
    def __init__(self, interface: vampnet.interface.Interface):
        self.interface = interface
        self.client = SimpleUDPClient(ip, s_port)

        self.timer = Timer()

        # interrupt
        self._interrupt = False

        # a params lock
        self._param_lock = Lock()
    

        # the 'knobs'
        # TODO: need a better way to store these
        # and set these. maybe a Param class is needed
        self.temperature: float = 1.0
        self.corrupt_time_amt: float = 0.1 # a combination of periodic propmt and droput
        self.corrupt_type: str = "periodic"
        self.corrupt_timbre: int = 1 # upper codebook mask
        self.seed: int = None # TODO: instrument should have saveable seed slots

        self.param_keys = (
            "temperature",
            "corrupt_time_amt",
            "corrupt_type",
            "corrupt_timbre",
            "seed",
        )


    def set_parameter(self, address: str, *args: list[any]) -> None:
        if address == "/temperature":
            self.temperature = args[0]
        elif address == "/corrupt_time_amt":
            self.corrupt_time_amt = args[0]
        elif address == "/corrupt_type":
            self.corrupt_type = args[0]
            assert self.corrupt_type in self.CORRUPT_TYPES, f"corrupt type must be one of {self.CORRUPT_TYPES}"
        elif address == "/corrupt_timbre":
            self.corrupt_timbre = args[0]
        elif address == "/seed":
            self.seed = args[0]
        else:
            print(f"Unknown address {address}")


    def process(self, address: str, *args) -> None:
        if address == "/process":
            # get the path to audio
            audio_path = Path(args[0])

            # make sure it exists, otherwise send an error message
            if not audio_path.exists():
                self.client.send_message("/error", f"File {audio_path} does not exist")
                return
            
            # load the audio
            sig = sn.read_from_file(audio_path)
            # sig = sn.transpose(sig, self.transpose)

            # run the vamp
            if self.is_vamping: 
                self._interrupt = True
                while self.is_vamping:
                    print(f"Waiting for the current process to finish")
                    time.sleep(0.05)
                print(f"Current process has finished, starting new process")
            
            sig = self.vamp(sig)

            # write the audio
            outpath = audio_path.with_suffix("_vamped.wav")
            sn.write(sig, outpath)

            # send a message that the process is done
            self.client.send_message("/done", f"File {audio_path} has been vamped")
            self.client.send_message("/process-result", str(outpath.resolve()))

        else:
            self.set_parameter(address, *args)
            

    def calculate_mask(self, z: torch.Tensor):
        with lock:
            # compute the periodic prompt, dropout amt
            assert not self.corrupt_type == "onsets", "onsets not supported yet"
            periodic_prompt = self.corrupt_time_amt if self.corrupt_type == "periodic" else 1
            dropout_amt = self.corrupt_time_amt if self.corrupt_type == "random" else 0.0
            upper_codebook_mask = self.corrupt_timbre

        mask = self.interface.build_mask(
            state.z_masked, 
            periodic_prompt=periodic_prompt,
            upper_codebook_mask=upper_codebook_mask,
            dropout_amt=dropout_amt, 
        )

        return mask


    @torch.inference_mode()
    def vamp(self, sig: sn.Signal):
        self.is_vamping = True

        cfg_guidance = None

        self.timer.tick("resample")
        sig = sn.resample(sig, self.interface.sample_rate)
        self.timer.tock("resample")

        # encode
        self.timer.tick("encode")
        codes = self.interface.encode(sig)
        self.timer.tock("encode")

        state = self.initialize_state(
            codes, 
            self.SAMPLING_STEPS, 
            cfg_guidance=cfg_guidance
        )

        timer.tick("sample")
        for _ in tqdm(range(self.SAMPLING_STEPS)):
            # seed
            torch.manual_seed(self.seed)

            # check for an interrupt 
            if self._interrupt:
                print(f'INTERRUPTED at step {i}')
                self._interrupt = False
                self.is_vamping = False
                return

            # build the mask
            mask = self.calculate_mask(state.z_masked)

            # apply mask
            state.z_masked = apply_mask(
                state.z_masked, mask, 
                self.interface.vn.mask_token
            )

            state = self.generate_step(
                state,
                temperature=self.temperature,
                mask_temperature=10.5,
                typical_filtering=False,
                typical_mass=0.15,
                typical_min_tokens=64,
                top_p=None,
                seed=self.seed,
                sample_cutoff=1.0,
                causal_weight=0.0,
                cfg_guidance=cfg_guidance
            )
        timer.tock("sample")

        z = state.z_masked[:state.nb] if cfg_guidance is not None else state.z_masked

        # decode
        timer.tick("decode")
        sig.wav = interface.decode(z)
        timer.tock("decode")

        self.is_vamping = False
        return sig
    

def process(address: str, *args: list[any]) -> None:
    system.process(address, *args)

def set_parameter(address: str, *args: list[any]) -> None:
    system.set_parameter(address, *args)

def start_server():
    dispatcher = Dispatcher()
    dispatcher.map("/process", process)

    dispatcher.map("/temperature", set_parameter)
    dispatcher.map("/corrupt_time_amt", set_parameter)
    dispatcher.map("/corrupt_type", set_parameter)
    dispatcher.map("/corrupt_timbre", set_parameter)
    dispatcher.map("/seed", set_parameter)

    server = ThreadingOSCUDPServer((ip, r_port), dispatcher)
    print(f"Serving on {server.server_address}")
    server.serve_forever()

interface = vampnet.interface.load_from_trainer_ckpt(
    ckpt_path="/home/hugo/soup/runs/debug/lightning_logs/version_49-best-new/checkpoints/vampnet-epoch=03-step=90141.ckpt", 
    codec_ckpt="/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"
)

system = VampNetDigitalInstrumentSystem(interface)

start_server()
