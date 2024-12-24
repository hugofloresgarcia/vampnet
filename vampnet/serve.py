from threading import Lock
import time
from dataclasses import dataclass
from pathlib import Path

from pythonosc.osc_server import ThreadingOSCUDPServer, BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

import torch
import vampnet
import vampnet.signal as sn
from vampnet.mask import apply_mask
from tqdm import tqdm

ip = "localhost"
s_port = 8002
r_port = 8001
DEVICE="mps"

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
        self.interface.vn = torch.compile(self.interface.vn)
        print(f"will send to {ip}:{s_port}")
        self.client = SimpleUDPClient(ip, s_port)

        self.timer = Timer()

        # interrupt
        self._interrupt = False
        self.is_vamping = False

        # a params lock
        self._param_lock = Lock()
    

        # the 'knobs'
        # TODO: need a better way to store these
        # and set these. maybe a Param class is needed
        self.temperature: float = 1.0
        self.corrupt_time_amt: float = 0.1 # a combination of periodic propmt and droput
        self.corrupt_type: str = "periodic"
        self.corrupt_timbre: int = 4 # upper codebook mask
        self.seed: int = -1 # TODO: instrument should have saveable seed slots

        self.param_keys = (
            "temperature",
            "corrupt_time_amt",
            "corrupt_type",
            "corrupt_timbre",
            "seed",
        )

        # dispatch a hello message
        print(f"sending hello message...")
        self.client.send_message("/hello",["Hello from VampNet"])
        self.client.send_message("/hello", [6., -2.])



    def set_parameter(self, address: str, *args: list[any]) -> None:
        with self._param_lock:
            print(f"Setting parameter {address} to {args}")
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
        timer = self.timer
        timer.tick("process")

        if address == "/process":
            print(f"Processing {address} with args {args}")
            # get the path to audio
            audio_path = Path(args[0])
            # patch 
            audio_path = Path("../") / audio_path

            # make sure it exists, otherwise send an error message
            if not audio_path.exists():
                print(f"File {audio_path} does not exist")
                self.client.send_message("/error", f"File {audio_path} does not exist")
                return
            
            # load the audio
            sig = sn.read_from_file(audio_path)
            sig = sig.to(DEVICE)
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
            outpath = audio_path.with_suffix(".vamped.wav")
            sn.write(sig, outpath)

            # send a message that the process is done
            self.client.send_message("/done", f"File {audio_path} has been vamped")
            self.client.send_message("/process-result", str(outpath.resolve()))

        else:
            print(f"PROC: Unknown address {address}")
        timer.tock("process")
            

    def calculate_mask(self, z: torch.Tensor):
        with self._param_lock:
            # compute the periodic prompt, dropout amt
            assert not self.corrupt_type == "onsets", "onsets not supported yet"
            # periodic_prompt = self.corrupt_time_amt if self.corrupt_type == "periodic" else 1
            if self.corrupt_type == "periodic":
                periods = [3, 5, 7, 11, 13, 17, 21, 0]
                # map corrupt_time_amt to a periodic prompt
                periodic_prompt = periods[int(self.corrupt_time_amt * (len(periods) - 1))]
                print(f"periodic prompt: {periodic_prompt}")
            dropout_amt = self.corrupt_time_amt if self.corrupt_type == "random" else 0.0
            upper_codebook_mask = self.corrupt_timbre

        mask = self.interface.build_mask(
            z, 
            periodic_prompt=periodic_prompt,
            upper_codebook_mask=upper_codebook_mask,
            dropout_amt=dropout_amt, 
        )

        return mask


    @torch.inference_mode()
    def vamp(self, sig: sn.Signal):
        self.is_vamping = True

        cfg_guidance = None
        timer = self.timer

        timer.tick("resample")
        sig = sn.resample(sig, self.interface.sample_rate)
        timer.tock("resample")

        # encode
        timer.tick("encode")
        codes = self.interface.encode(sig.wav)
        timer.tock("encode")

        state = self.interface.vn.initialize_state(
            codes, 
            self.SAMPLING_STEPS, 
            cfg_guidance=cfg_guidance
        )

        # seed
        if self.seed >= 0:
            torch.manual_seed(self.seed)
        else:
            torch.manual_seed(int(time.time()))

        timer.tick("sample")
        for _ in tqdm(range(self.SAMPLING_STEPS)):

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

            state = self.interface.vn.generate_step(
                state,
                temperature=self.temperature,
                mask_temperature=10.5,
                typical_filtering=False,
                typical_mass=0.15,
                typical_min_tokens=64,
                top_p=None,
                sample_cutoff=1.0,
                # causal_weight=0.0,
                # cfg_guidance=cfg_guidance
            )
        timer.tock("sample")

        z = state.z_masked[:state.nb] if cfg_guidance is not None else state.z_masked

        # decode
        timer.tick("decode")
        sig.wav = interface.decode(z)
        timer.tock("decode")

        self.is_vamping = False
        return sig
    


def start_server():
    dispatcher = Dispatcher()
    dispatcher.map("/process", system.process)

    dispatcher.map("/temperature", system.set_parameter)
    dispatcher.map("/corrupt_time_amt", system.set_parameter)
    dispatcher.map("/corrupt_type", system.set_parameter)
    dispatcher.map("/corrupt_timbre", system.set_parameter)
    dispatcher.map("/seed", system.set_parameter)
    dispatcher.set_default_handler(lambda a, *r: print(a, r))

    server = BlockingOSCUDPServer((ip, r_port), dispatcher)
    print(f"Serving on {server.server_address}")
    server.serve_forever()

interface = vampnet.interface.load_from_trainer_ckpt(
    ckpt_path="ckpt-vctk.ckpt", 
    codec_ckpt="/Users/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"
)

system = VampNetDigitalInstrumentSystem(interface)

system.interface.to(DEVICE)

start_server()
