from threading import Lock
import time
from dataclasses import dataclass
from pathlib import Path
import argbind

from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

import torch
import vampnet
import vampnet.signal as sn
from vampnet.mask import apply_mask
from vampnet.train import VampNetTrainer
from tqdm import tqdm


@dataclass
class Param:
    name: str
    value: any
    param_type: type
    range: tuple = None  # Optional range for validation

    def set_value(self, new_value):
        if self.range and not (self.range[0] <= new_value <= self.range[1]):
            raise ValueError(f"Value {new_value} for {self.name} is out of range {self.range}")
        self.value = self.param_type(new_value)


class ParamManager:
    def __init__(self):
        self._params = {}
        self._lock = Lock()

    def register_param(self, name, initial_value, param_type, range=None):
        with self._lock:
            if name in self._params:
                raise ValueError(f"Parameter {name} already registered")
            self._params[name] = Param(name, initial_value, param_type, range)

    def set_param(self, name, value):
        with self._lock:
            if name not in self._params:
                raise ValueError(f"Parameter {name} not registered")
            self._params[name].set_value(value)

    def get_param(self, name):
        with self._lock:
            if name not in self._params:
                raise ValueError(f"Parameter {name} not registered")
            return self._params[name].value

    def list_params(self):
        with self._lock:
            return {name: param.value for name, param in self._params.items()}


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
    SAMPLING_STEPS = 12
    
    def __init__(self, 
        interface: vampnet.interface.Interface, 
        ip: str,
        s_port: int, r_port: int,
        device: str
    ):
        self.ip = ip
        self.s_port = s_port
        self.r_port = r_port
        self.device = device

        # set up interface
        self.interface = interface
        self.interface.to(device)

        self.interface.vn = torch.compile(self.interface.vn)

        print(f"will send to {ip}:{s_port}")
        self.client = SimpleUDPClient(ip, s_port)

        # set up timer
        self.timer = Timer()

        # interrupts
        self._interrupt = False
        self.is_vamping = False

        # a lock for the parameters
        self._param_lock = Lock()
    
        # the 'knobs'
        # TODO: need a better way to store these
        # and set these. maybe a Param class is needed
        self.temperature: float = 1.0
        self.periodic_tokens: int = 3 # a combination of periodic propmt and droput
        self.periodic_rms: int = 3 # a combination of periodic propmt and droput
        self.upper_codebook_mask: int = 4 # upper codebook mask
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
            elif address == "/corrupt_timbre":
                self.corrupt_timbre = int(args[0])
            elif address == "/seed":
                self.seed = args[0]
            else:
                print(f"Unknown address {address}")
            
    def calculate_mask(self, z: torch.Tensor):
        raise NotImplementedError()

        mask = self.interface.build_mask(
            z, 
            periodic_prompt=periodic_prompt,
            upper_codebook_mask=upper_codebook_mask,
            dropout_amt=dropout_amt, 
        )

        # save the mask as a txt file (numpy)
        import numpy as np
        np.savetxt("mask.txt", mask[0].cpu().numpy(), fmt="%d")

        return mask


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
            sig = sig.to(self.interface.device)

            # run the vamp
            if self.is_vamping: 
                print(f"interrupting!")
                self._interrupt = True
                self.interface.vn.interrupt = True
                while self.is_vamping:
                    # print(f"Waiting for the current process to finish")
                    time.sleep(0.05)
                print(f"interrupted!")
                self.interface.vn.interrupt = False
                self._interrupt = False
                print(f"Current process has finished, starting new process")
            
            sig = self.vamp(sig)
            if sig is None: # we were interrupted
                return

            # write the audio
            outpath = audio_path.with_suffix(".vamped.wav")
            sn.write(sig, outpath)

            # send a message that the process is done
            self.client.send_message("/done", f"File {audio_path} has been vamped")
            self.client.send_message("/process-result", str(outpath.resolve()))

        else:
            print(f"PROC: Unknown address {address}")
        timer.tock("process")
            

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

        timer.tick("sample")
        # build the mask
        # MUST APPLY MASK BEFORE INITIALIZING STATE
        mask = self.calculate_mask(codes)
        codes = apply_mask(
            codes, mask, 
            self.interface.vn.mask_token
        )

        # seed
        if self.seed >= 0:
            torch.manual_seed(self.seed)
        else:
            torch.manual_seed(int(time.time()))

        z = self.interface.vn.generate(
            codes=codes, 
        )
        if z is None:
            # we were interrupted
            self.is_vamping = False
            return None
        timer.tock("sample")

        # decode
        timer.tick("decode")
        sig.wav = interface.decode(z)
        timer.tock("decode")

        self.is_vamping = False
        return sig
    

    def start_server(self,):
        dispatcher = Dispatcher()
        dispatcher.map("/process", self.process)

        dispatcher.map("/temperature", self.set_parameter)
        dispatcher.map("/corrupt_time_amt", self.set_parameter)
        dispatcher.map("/corrupt_type", self.set_parameter)
        dispatcher.map("/corrupt_timbre", self.set_parameter)
        dispatcher.map("/seed", self.set_parameter)
        dispatcher.set_default_handler(lambda a, *r: print(a, r))

        server = ThreadingOSCUDPServer((self.ip, self.r_port), dispatcher)
        print(f"Serving on {server.server_address}")
        server.serve_forever()


@argbind.bind()
def main(ip = "localhost", s_port = 8002, r_port = 8001, 
         device="mps", ckpt: str = "hugggof/vampnetv2-mode-vampnet_rms-latest"):

    bundle = VampNetTrainer.from_pretrained(ckpt)

    interface = vampnet.interface.Interface(
        bundle.codec, 
        bundle.model, 
        bundle.controller
    )

    system = VampNetDigitalInstrumentSystem(
        interface, ip=ip, s_port=s_port,
        r_port=r_port, device=device
    )

    system.start_server()

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()
