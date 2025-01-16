from threading import Lock
import time
from dataclasses import dataclass
from pathlib import Path
import argbind

from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from gradio_client import Client, handle_file

import torch
import vampnet
import vampnet.dsp.signal as sn
from vampnet.mask import apply_mask
from vampnet.train import VampNetTrainer
from vampnet.util import Timer
from tqdm import tqdm

timer = vampnet.util.Timer()


@dataclass
class Param:
    name: str
    value: any
    param_type: type
    range: tuple = None  # Optional range for validation
    step: float = 0.1

    def set_value(self, new_value):
        if self.range and not (self.range[0] <= new_value <= self.range[1]):
            raise ValueError(f"Value {new_value} for {self.name} is out of range {self.range}")
        # quantize to step
        new_value = round(new_value / self.step) * self.step
        self.value = self.param_type(new_value)


class ParamManager:
    def __init__(self):
        self._params = {}
        self._lock = Lock()

    def register(self, name, initial_value, param_type, range=None, **kwargs):
        with self._lock:
            if name in self._params:
                raise ValueError(f"Parameter {name} already registered")
            self._params[name] = Param(name, initial_value, param_type, range, **kwargs)

    def set(self, name, value):
        with self._lock:
            if name not in self._params:
                raise ValueError(f"Parameter {name} not registered")
            self._params[name].set_value(value)

    def get(self, name):
        with self._lock:
            if name not in self._params:
                raise ValueError(f"Parameter {name} not registered")
            return self._params[name].value

    def asdict(self):
        with self._lock:
            return dict(self._params)

    def list(self):
        with self._lock:
            return {name: param.value for name, param in self._params.items()}

def create_param_manager():
    pm = ParamManager()
    pm.register("seed", -1, int, step=1)
    pm.register("temperature", 1.0, float, (0.5, 10.0), step=0.01)
    pm.register("controls_periodic_prompt", 5, int, (0, 100), step=1)
    pm.register("controls_drop_amt", 0.15, float, (0.0, 1.0), step=0.01)
    pm.register("codes_periodic_prompt", 32, int, (0, 100), step=1)
    pm.register("codes_upper_codebook_mask", 1, int, (0, 10), step=1)
    pm.register("mask_temperature", 1000.0, float, (0.1, 100000.0), step=0.1)
    pm.register("typical_mass", 0.15, float, (0.0, 1.0), step=0.01)
    return pm

class VampNetOSCManager:

    def __init__(
        self, 
        ip: str, 
        s_port: str, 
        r_port: str,
        process_fn: callable
    ):
        self.ip = ip
        self.s_port = s_port
        self.r_port = r_port

        # register parameters
        self.pm = create_param_manager()

        # register the process_fn
        self.process_fn = process_fn

        print(f"will send to {ip}:{s_port}")
        self.client = SimpleUDPClient(ip, s_port)

        # dispatch a hello message
        print(f"sending hello message...")
        self.client.send_message("/hello",["Hello from VampNet"])
        self.client.send_message("/hello", [6., -2.])


    def start_server(self,):
        dispatcher = Dispatcher()
        dispatcher.map("/process", self.process_fn)

        # connect params from manager to dispatcher
        for param_name in self.pm.list().keys():
            dispatcher.map(f"/{param_name}", self._osc_set_param(param_name))

        dispatcher.map("/get_params", self._osc_get_params)

        dispatcher.set_default_handler(lambda a, *r: print(a, r))

        server = ThreadingOSCUDPServer((self.ip, self.r_port), dispatcher)
        print(f"Serving on {server.server_address}")
        server.serve_forever()

    def _osc_set_param(self, param_name):
        def handler(_, *args):
            try:
                self.pm.set(param_name, args[0])
                print(f"Set {param_name} to {self.pm.get(param_name)}")
            except ValueError as e:
                print(f"Error setting parameter {param_name}: {e}")
        return handler

    def _osc_get_params(self, address, *args):
        param_names = list(self.pm.list().keys())
        print(f"Returning parameter names: {param_names}")
        self.client.send_message("/params", ",".join(param_names))

    def done(self, outpath: str):
        # send a message that the process is done
        self.client.send_message("/done", f"File {outpath} has been vamped")
        self.client.send_message("/process-result", outpath)

    def error(self, msg: str):
        self.client.send_message("/error", msg)
        

# a class that wraps the vampnet interface
# and provides a way to interact with it
# over OSC with digital musical instruments
class VampNetDigitalInstrumentSystem:
    SAMPLING_STEPS = 16
    
    def __init__(self, 
        interface: vampnet.interface.Interface, 
        ip: str,
        s_port: int, r_port: int,
        device: str
    ):
        self.osc_manager = VampNetOSCManager(
            ip=ip, s_port=s_port, r_port=r_port, 
            process_fn=self.process
        )
        self.device = device

        # set up interface
        self.interface = interface
        self.interface.to(device)

        # set up timer
        self.timer = Timer()

        # interrupts
        self._interrupt = False
        self.is_vamping = False

        self.pm = self.osc_manager.pm

    def process(self, address: str, *args) -> None:
        timer = self.timer
        timer.tick("process")

        if address == "/process":
            print(f"Processing {address} with args {args}")
            # get the path to audio
            audio_path = Path(args[0])
            # TODO: FIXME: patch 
            audio_path = Path("pd/") / audio_path

            # make sure it exists, otherwise send an error message
            if not audio_path.exists():
                print(f"File {audio_path} does not exist")
                self.osc_manager.error(f"File {audio_path} does not exist")
                return
            
            # load the audio
            sig = sn.read_from_file(audio_path)
            print(f"got a file with duration {sig.duration}")
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
            # delete the input buf
            
            Path(audio_path).unlink()

            self.osc_manager.done(str(outpath.resolve()))
        else:
            print(f"PROC: Unknown address {address}")
        timer.tock("process")

    # this belongs to the gradio version. 
    @torch.inference_mode()
    def vamp(self, sig: sn.Signal):
        seed = self.pm.get("seed")
        self.is_vamping = True

        timer = self.timer

        timer.tick("preprocess")
        sig = sig.cpu()
        ldns = sn.loudness(sig)
        sig = self.interface.preprocess(sig)    
        timer.tock("preprocess")

        # controls
        timer.tick("controls")
        ctrls = self.interface.controller.extract(sig)
        onset_idxs = sn.onsets(sig, hop_length=self.interface.codec.hop_length)
        # ctrl_masks = self.interface.build_ctrl_masks(
        #     ctrls, 
        #     periodic_prompt=self.pm.get("controls_periodic_prompt")
        # )
        ctrl_masks = {}
        for k,ctrl in ctrls.items():
            ctrl_masks[k] = self.interface.rms_mask(
                ctrl, onset_idxs, periodic_prompt=self.pm.get("controls_periodic_prompt")
            )
        if "hchroma-12c-top2" in ctrls:
            print("disabling hchroma")
            ctrl_masks["hchroma-12c-top2"] = torch.zeros_like(ctrl_masks["hchroma-12c-top2"])

        # encode
        timer.tick("encode")
        codes = self.interface.encode(sig.wav)
        timer.tock("encode")

        # build the mask
        mask = self.interface.build_codes_mask(
            codes, 
            periodic_prompt=self.pm.get("codes_periodic_prompt"),
            upper_codebook_mask=self.pm.get("codes_upper_codebook_mask")
        )
        codes = apply_mask(codes, mask, self.interface.vn.mask_token)

        # seed
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(int(time.time()))

        timer.tick("sample")
        z = self.interface.vn.generate(
            codes=codes, 
            ctrls=ctrls,
            ctrl_masks=ctrl_masks,
            temperature=self.pm.get("temperature"),
            mask_temperature=self.pm.get("mask_temperature"),
            typical_filtering=True,
            typical_mass=self.pm.get("typical_mass"),
            sampling_steps=self.SAMPLING_STEPS,
            debug=False
        )

        if z is None:
            # we were interrupted
            self.is_vamping = False
            return None
        timer.tock("sample")

        # decode
        timer.tick("decode")
        sig.wav = self.interface.decode(z).cpu()
        sig = sn.normalize(sig, ldns)
        timer.tock("decode")

        self.is_vamping = False
        return sig
    

class GradioVampNetSystem:

    def __init__(self, 
        url: str,
        ip: str,
        s_port: int, r_port: int,
    ):
        self.osc_manager = VampNetOSCManager(
            ip=ip, s_port=s_port, r_port=r_port, 
            process_fn=self.process
        )
        self.pm = self.osc_manager.pm
        
        # TODO: cross check API versions with the osc manager!!!
        self.client = Client(src=url, download_files=".gradio")

    
    def process(self, address: str, *args):
        if address != "/process":
            raise ValueError(f"Unknown address {address}")

        print(f"Processing {address} with args {args}")
        # get the path to audio
        audio_path = Path(args[0])
        # TODO: FIXME: patch 
        audio_path = Path("pd/") / audio_path

        # make sure it exists, otherwise send an error message
        if not audio_path.exists():
            print(f"File {audio_path} does not exist")
            self.osc_manager.error(f"File {audio_path} does not exist")
            return

        timer.tick("predict")
        outpath = self.client.predict(
            # TODO: the parameters should actually be part of a dataclass now that i think about it. 
            data=handle_file(audio_path),
            param_1=None,
            # param_1=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
            param_2=self.pm.get("seed"),
            param_3=self.pm.get("temperature"),
            param_4=self.pm.get("controls_periodic_prompt"),
            param_6=self.pm.get("controls_drop_amt"),
            param_7=self.pm.get("codes_periodic_prompt"),
            param_8=self.pm.get("codes_upper_codebook_mask"),
            param_9=self.pm.get("mask_temperature"),
            param_10=self.pm.get("typical_mass"),
            api_name="/api-vamp"
        )

        timer.tock("predict")
        self.osc_manager.done(outpath)


@argbind.bind(without_prefix=True)
def main(ip = "localhost", s_port = 8002, r_port = 8001, 
         device="cpu", ckpt: str = "hugggof/vampnetv2-mode-vampnet_rms-latest"):

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

    system.osc_manager.start_server()


@argbind.bind(without_prefix=True)
def gradio_main(url: str="http://localhost:7860/"):

    system = GradioVampNetSystem(
        url="http://localhost:7860/",
        ip="localhost", s_port=8002, r_port=8001,
    )

    system.osc_manager.start_server()


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        gradio_main()
