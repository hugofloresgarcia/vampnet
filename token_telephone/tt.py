from ttutil import hsv_to_rgb, dbg, log, set_debug, pow2db, db2pow
from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import time
from threading import Thread
import gc
gc.disable()

import sounddevice as sd

from blessed import Terminal

import numpy as np
import torch
from einops import rearrange

PROFILE = False
DEBUG = False
DEBUG_NO_VAMPNET = False
set_debug(DEBUG)
# if DEBUG:
#     import gc
#     # log when gc start and stops
#     gc.set_debug(gc.DEBUG_STATS)

@dataclass 
class LoadState:
    t0: float = None
    loaded: bool = False

load_state = LoadState()    

def on_random_color():
    def random_rgb_bg():
        return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
    return term.on_color_rgb(*random_rgb_bg())

# draw the intro screen before slow imports
def color_tokenize_txt(text: str):
    # apply a random bg color to each letter
    return "".join(on_random_color()(letter) for letter in text)

def color_tokenize_words(text: str):
    return " ".join(on_random_color()(word) for word in text.split(" "))

def draw_intro_screen():
    global load_state
    load_state.t0 = time.time()
    avg_time = 20 # average loading time

    while not load_state.loaded: 
        print(term.clear)
        print(term.move_xy(0, 1) + term.center(color_tokenize_words("hugo flores garcía")))
        print(term.move_xy(0, 3) + term.center(color_tokenize_words("and")))
        print(term.move_xy(0, 5) + term.center(color_tokenize_words("stephan moore")))
        print(term.move_xy(0, 7) + term.center(color_tokenize_words("present")))
        print(term.move_xy(0, 9) + term.center(term.bold(color_tokenize_txt("token telephone"))))

        # print(term.move_xy(0, 10) + term.center(color_tokenize_txt("loading ")), end="")
        # make a little loading bar
        elapsed = time.time() - load_state.t0
        num_dots = int((elapsed / avg_time) * 20)
        num_spaces = 20 - num_dots
        print(term.move_xy(0, 12) + term.center(color_tokenize_words("loading")))
        print(term.move_xy(0, 13) + term.center(color_tokenize_txt(f"[{'.' * num_dots}") + f"{' ' * num_spaces}]"))
        time.sleep(0.3)
    
    log(f"loading took {time.time() - load_state.t0} seconds")
    return
    
# the program
term = Terminal()

# draw the intro screen on a background thread
Thread(target=draw_intro_screen).start()

# disable garbage collection
from audiotools import AudioSignal
from vamp_helper import load_interface, ez_variation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# ~~~~~~  configs!     ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MAX_LOUDNESS = -20
MIN_LOUDNESS = -40
COLS = 40
ROWS = 13

device = 'Scarlett 4i4 4th Gen'
sample_rate = 48000
num_channels = 4
blocksize = 16384


# TODO: 
# still some quirks to work around recording time: 
# do we wanna stop recording and wait a full cycle before letting people record again? 
# how do we wanna balance the volume of a new input vs what's currently gonig on?
# should people have to take turns in between new loops? 
# otherwise, we're doing great i think
# we also need to add a crossfade. This means maybe cutting off the last 0.1 seconds of the loop, and the beginning 0.1
# and use that to crossfade.

# TODO: do I wanna train a diff model to swap every 2hrs or something? 
# how lond does model swapping take?  how can I make it faster?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~  looper      ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass 
class State:
    # looper state
    feedback: float = 0.25
    duration: float = 5.0
    record_channel: int = 0

    loopbuf: np.ndarray = None # the main loop buffer. the token telephone audio is here
    looper_in: np.ndarray = None # a buffer that stores the audio that's being recorded

    buf_in: np.ndarray = None # the input block with audio samples in the audio callbac
    lookback_buf: np.ndarray = None # stores some lookback audio for when the threshold is passed, to propery capture transients

    recording: bool = False
    playing: bool = False

    # ramps
    record_ramp_in: bool = False
    record_ramp_out: bool = False

    # n_record_layers: int = 2 # number of times we'll record over before clearing
    # cur_rec_layer: int = 0
    recording_locked: bool = False

    rec_time: float = 0
    cur_hold_time: float = None
    pos: int = 0
    rms_db: float = float("-inf")

    trig_threshold_db = -25 # a more sane default is -20
    hold_seconds = 1.0
    rel_threshold_db = -40 # a more sane default is -30

    status: str = field(default=None)

    # token telephone configs
    z_buf: torch.Tensor = None
    input_ready = False
    input_channel = 0
    token_telephone_processing: bool = False
    num_telephone_chans = 4
    tt_cur_ch = 0

    def __post_init__(self):
        self.loopbuf = np.zeros((num_channels, int(self.duration * sample_rate)))
        self.looper_in = np.zeros((1, int(self.duration * sample_rate)))

        # hold 200ms of lookback to account for rising attacks. 
        num_lookback_samples = max(int(sample_rate * 0.2), int(blocksize))
        log(f"num_lookback_samples {num_lookback_samples} ({num_lookback_samples / sample_rate} seconds)")
        self.lookback_buf = np.zeros((1, num_lookback_samples)) 
        
        self.buf_in = np.zeros((num_channels, blocksize))



def check_if_record(st: State, ain: np.ndarray, on_release_callback=None):
    # get our rms value
    rms = pow2db(np.sqrt(np.mean(ain**2)))
    st.rms_db = rms

    # determine if we should ater the looper state
    # if we werent recording and we cross the trigger threshold
    # start recording
    # if not st.recording and rms > st.trig_threshold_db and not st.recording_locked:
    if not st.recording and rms > st.trig_threshold_db and not st.recording_locked:
        st.recording = True
        st.record_ramp_in = True

    # if we were recording and we cross the release threshold
    # begin the hold period
    if (st.recording and rms < st.rel_threshold_db) or st.rec_time > (st.duration-st.hold_seconds):
        # if we dont have a hold time, set it
        if st.cur_hold_time is None:
            st.cur_hold_time = time.time()

        # release if we have a hold time and we've held for the required time, 
        if (time.time() - st.cur_hold_time) > st.hold_seconds:
            st.record_ramp_out = True
            st.rec_time = 0
            if on_release_callback is not None:
                st.input_ready = True
                on_release_callback(st)
            st.cur_hold_time = None
        else:
            pass
    else:
        st.cur_hold_time = None


def launch_token_telephone(st: State):
    if interface is None:
        log("no interface loaded, can't do token telephone!")
        time.sleep(10)
        return

    # if we're already processing, do nothing
    if st.token_telephone_processing:
        return 
    else:
        log("starting token telephone!")
        Thread(target=do_token_telephone, args=(st,)).start()


def do_token_telephone(st: State,):
    st.token_telephone_processing = True
    while True:
        lrc = st.record_channel
        t0 = time.time()
        cur_ch = st.tt_cur_ch

        # if there was input ready, start back from the top. 
        if st.input_ready:
            log(f"there was input ready, processing!")
            # NOTE: hugo, trying something new here. what happens if
            # we don't reset the channel when input is ready, 
            # and instead let it come in anywhere in the cycle?
            # st.tt_cur_ch = 0 # uncomment to go back to reality

            # clear the lrc, reset for next record. 
            st.input_ready = False

            # reocrd the channel that we'll be processing in and lock recording
            st.input_channel = cur_ch
            st.recording_locked = True

            # first, let's preprocess looper in
            sig_looper_in = AudioSignal(
                torch.from_numpy(st.looper_in).unsqueeze(0),
                sample_rate=sample_rate
            )
            sig_loopbuf_curch = AudioSignal(
                torch.from_numpy(st.loopbuf[cur_ch:cur_ch+1]).unsqueeze(0),
                sample_rate=sample_rate
            )
            # make sure looperin matches the midpoint in loudness
            ldns_mid = max(sig_loopbuf_curch.loudness(), sig_looper_in.loudness())
            sig_looper_in = sig_looper_in.normalize(ldns_mid)
            st.looper_in = sig_looper_in.samples.cpu().numpy().squeeze(0)

            st.loopbuf[cur_ch:cur_ch + 1] = (
                st.looper_in + st.loopbuf[cur_ch:cur_ch+1] * st.feedback
            )
            # also lower the volumes of the other channels
            for i in range(4):
                if i != cur_ch:
                    st.loopbuf[i:i+1] = st.loopbuf[i:i+1] * 0.5 # -3dB

            st.looper_in = np.zeros_like(st.looper_in)

        loop_input = st.loopbuf[cur_ch:cur_ch+1]

        # ~~~ VAMPNET STUFF ~~~~
        sig = AudioSignal(
            torch.from_numpy(loop_input).unsqueeze(0),
            sample_rate=sample_rate
        )
        input_loudness = sig.loudness()
        log(f"INPUT loudness {input_loudness}")
        if input_loudness > MAX_LOUDNESS:
            log(f"input loudness {input_loudness} is over {MAX_LOUDNESS}!")
            sig = sig.normalize(MAX_LOUDNESS)
        elif input_loudness < MIN_LOUDNESS:
            log(f"input loudness {input_loudness} is under {MIN_LOUDNESS}!")
            sig = sig.normalize(MIN_LOUDNESS)

        sig = ez_variation(interface, sig)
        sig = sig.resample(sample_rate)

        # notify if we've gone over the loudness
        sig = sig.normalize(input_loudness)
        outloudness = sig.loudness()
        if outloudness > MAX_LOUDNESS:
            log(f"out loudness {sig.loudness()} is over {MAX_LOUDNESS}!")
            sig = sig.normalize(MAX_LOUDNESS)
        elif outloudness < MIN_LOUDNESS:
            log(f"out loudness {sig.loudness()} is under {MIN_LOUDNESS}!")
            sig = sig.normalize(MIN_LOUDNESS)

        # put it back in the loopbuf
        # write to the next channel
        # (TODO: instead of trimming to loopbuf.shape[1], maybe we can just have the loopbuf be the right size from init time.)
        cur_ch = (cur_ch + 1) % st.num_telephone_chans
        st.tt_cur_ch = cur_ch
        if False: # HUGO: is there a time where we want feedback? 
            st.loopbuf[cur_ch:cur_ch+1] = (
                sig.samples.cpu().numpy().squeeze(0)[:, :st.loopbuf.shape[1]]
                + st.feedback * st.loopbuf[cur_ch:cur_ch+1]
            )
        else:
            st.loopbuf[cur_ch:cur_ch+1] = (
                sig.samples.cpu().numpy().squeeze(0)[:, :st.loopbuf.shape[1]]
            )

        log(f"output loudness {sig.loudness()}")
        log(f"telephone loop took {time.time() - t0} seconds... next channel {cur_ch}\n\n")

        # if we've made it back to the input channel, we can unlock the recording
        log(f"cur_ch {cur_ch} input_channel {st.input_channel}")
        if cur_ch == st.input_channel:
            st.recording_locked = False
            log(f"recording unlocked!")


        # unlock the recording if we've successfully written to all channels
        # if st.recording_locked and cur_ch == 0:
        #     st.recording_locked = False
        #     log(f"recording locked {st.recording_locked}")

    st.token_telephone_processing = False
    return

# TODO: since we're using this really high threshold
# we always need to record about 100ms in advance, to catch the beginning of the attacks. 

def looper_process_block(st, block: np.ndarray):
    lrc = st.record_channel

    # treat the lookback buffer as a circular buffer
    st.lookback_buf = np.roll(st.lookback_buf, block.shape[1], axis=1)
    st.lookback_buf[:, -block.shape[1]:] = block[lrc:lrc+1, :]


    # check if we need to record.
    if st.recording:
        start_i = (st.pos + block.shape[1]) - st.lookback_buf.shape[1]
        end_i = st.pos + st.lookback_buf.shape[1]

        indices = np.take(
            np.arange(st.loopbuf.shape[1]),
            np.arange(start_i, end_i),
            mode="wrap"
        )
        _audio_in = st.lookback_buf[:, :]
        # ramp in if we need to
        if st.record_ramp_in:
            _audio_in = _audio_in * np.linspace(0, 1, _audio_in.shape[1])
            st.record_ramp_in=False
        
        if st.record_ramp_out:
            _audio_in = _audio_in * np.linspace(1, 0, _audio_in.shape[1])
            st.record_ramp_out=False
            st.recording = False

        st.looper_in[:, indices] = (
            0.9 * st.looper_in[:, indices] + _audio_in
        )

        # incremement the recording time
        st.rec_time += st.lookback_buf.shape[1] / sample_rate
    
    # check if we need to play
    crossfade_samples = int(0.1 * sample_rate)
    if st.playing:
        play_pos = (st.pos + block.shape[1]) % st.loopbuf.shape[1] # read one buffer ahead
        indices = np.arange(play_pos, play_pos + block.shape[1]) 
        block = st.loopbuf.take(indices, axis=1, mode="wrap")[:, :] # this doesn't have any crossfading. # TODO: this is still not working! 

    # if we've recorded more than the loop size
    if st.rec_time > st.duration and st.recording:
        # play the loop
        play_pos = st.pos + block.shape[1] # read one buffer ahead
        indices = np.arange(play_pos, play_pos + block.shape[1])

        block[lrc:lrc] = st.looper_in.take(indices, axis=1, mode="wrap")[:, :]
    
    # advance looper state
    st.pos = (st.pos + block.shape[1]) % st.loopbuf.shape[1]    
    
    return block

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~  drawing      ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_rms_bar(st, x, y, width, height):
    rms_min = -50
    rms_max = -10
    rms = st.rms_db
    rms = max(rms, rms_min)
    threshold = st.trig_threshold_db
    rel_threshold = st.rel_threshold_db

    rms_block = int((rms - rms_min) / (rms_max - rms_min) * height)
    threshold_block = (threshold - rms_min) / (rms_max - rms_min) * height
    rel_threshold_block = (rel_threshold - rms_min) / (rms_max - rms_min) * height

    # draw the rms curve
    for i in range(rms_block, height+4):
        with term.location(x+4, y+height-i):
            print(term.clear_bol)
    for i in range(rms_block):
        rms_val = i * (rms_max - rms_min) / height + rms_min
        with term.location(x, y+height-2-i):
            if i < threshold_block:
                print("   " + term.on_green(f"*"))
            else:
                print("   " + term.on_red(f"*"))

    # at the very bottom of the bar, draw the rms value
    with term.location(x, y+height-1):
        print(f"{rms:.1f}dB")
        # print(f"  rms")


def draw_looper(st):
    x = 0
    y = 0
    width = COLS
    height = ROWS

    tt_refresh_every = 0.3
    if not hasattr(draw_looper, "last_draw"):
        draw_looper.last_draw = 0
        should_draw = True
    else:
        should_draw = (time.time() - draw_looper.last_draw) > tt_refresh_every
        if should_draw:
            draw_looper.last_draw = time.time()


    draw_rms_bar(st, x, y, width - 10, height)
        
    if should_draw:
        with term.location(width // 2-4, 1):
            for i, letter in enumerate("token telephone"):
                print(on_random_color()(letter), end="")

    # with term.location(ROWS-2, COLS // 2):
        # print(f"status {st.status}!!!")


    # if we're recording, draw a red unlderlined "rec" sign on the bottom right
    # with term.location(width-8, height-1):
    #     if st.recording:
    #         print(term.on_red("rec"))
    #     else:
    #         print(term.on_gray50("rec"))
    
    # # if we're playing draw a green underline "play" sign on the bottom right
    # with term.location(width-4, height-1):
    #     if st.playing:
    #         print(term.on_green("play"))
    #     else:
    #         print(term.on_gray50("play"))


    # draw the timeline at the bottom using ---
    with term.location(6, height):
        timeline = ["-"] * (width - 12)
        playhead = int((st.pos / st.loopbuf.shape[1]) * (width - 12))
        timeline[playhead] = "v"
        print("|"+"".join(timeline) + "|")

    
    # draw the main message at the very center:
    msg_loc = (width // 2, height // 2+1)
    _x, _y = msg_loc
    if not st.recording:
        if not st.recording_locked:
            print(term.move_xy(0, _y-1) + term.center("make a sound", width=width+5))
            print(term.move_xy(0, _y+0) + term.center("to", width=width+5))
            print(term.move_xy(0, _y+1) + term.center("record", width=width+5))
        else:
            # how many seconds left until we can record again?
            # how many more chs do we need to go through before we can record again?
            if st.tt_cur_ch < st.input_channel:
                chs_remaining = st.input_channel - st.tt_cur_ch
            else:
                chs_remaining = 4-st.tt_cur_ch + st.input_channel
            locked_time_remaining = chs_remaining * st.duration + st.duration - (st.pos / sample_rate)
            print(term.move_xy(0, _y-1) + term.center("please wait", width=width+5))
            print(term.move_xy(0, _y+0) + term.center(term.on_green(f"{locked_time_remaining:.1f}s"), width=width+5))
            print(term.move_xy(0, _y+1) + term.center("for your turn :)", width=width+5))
    else: 
        print(term.move_xy(0, _y-1) + term.center(term.on_red("recording"), width=width+5))
        print(term.move_xy(0, _y+0) + term.center(f"{(st.duration) - st.rec_time:.1f}s left", width=width+5))
        print(term.move_xy(0, _y+1) + term.center("", width=width+5))


    # we'll draw channel 0 (1) on the bottom right corner
    # channel 1 (2) on the top right corner
    # channel 2 (3) on the top left corner
    # channel 3 (4) on the bottom left corner
    my = 3 # margin
    mx = 10
    locations = {
        1: (width - mx, height - my),
        2: (width - mx, 1+my),
        3: (mx, 1+my),
        4: (mx, height - my),
    }
    for i in range(1, 5):
        if should_draw:
            if st.tt_cur_ch == i - 1 and st.token_telephone_processing:
                x, y = locations[i]
                on_random_colors = lambda n: "".join(on_random_color()(" ") for _ in range(n))
                print(term.move_xy(x, y-1) + on_random_colors(5))
                print(term.move_xy(x, y) + on_random_color()(" ") + f" {i} " + on_random_color()(" "))
                print(term.move_xy(x, y+1) + on_random_colors(5))
            else:
                # same thing, but a gray instead of random colors
                x, y = locations[i]
                on_gray_colors = lambda n: "".join(term.on_gray50(" ") for _ in range(n))
                print(term.move_xy(x, y-1) + on_gray_colors(5))
                print(term.move_xy(x, y) + term.on_gray50(" ") + f" {i} " + term.on_gray50(" "))
                print(term.move_xy(x, y+1) + on_gray_colors(5))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~  live audio   ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def audio_init():
    sd.default.samplerate = sample_rate
    sd.default.device = device
            
# ~~~~~~ the main audio callback ~~~~~~~~~
def callback(st, indata, outdata, frames, _time, status):
    t0 = time.time()
    lrc = st.record_channel

    if status:
        log(f"status is {status}")
        st.status = status
        
    # log dtype, status, frames, time, max min
    # log(f"indata {indata.dtype} max {indata.max()} min {indata.min()} {status} {frames} {_time}")


    ain = rearrange(indata, 't n -> n t', n=num_channels)
    
    # convert audio to from int32 to float32
    ain = ain.astype(np.float32) / np.iinfo(np.int16).max
    buf_in = ain

    # if it's all zeros, we're not recording
    # so we can just pass it through
    if np.all(buf_in == 0):
        st.status = st.status + "no input"
        return 

    st.buf_in = buf_in
    check_if_record(
            st, buf_in, 
            on_release_callback=launch_token_telephone
    )
    buf_in = looper_process_block(st, buf_in)

    # pass our st.loopbuf to the output
    ain = buf_in

    # convert back to int32
    ain =  (ain * np.iinfo(np.int16).max).astype(np.int16)

    outdata[:] = rearrange(ain, 'n t -> t n')

    # log(f"outdata {outdata.dtype} max {outdata.max()} min {outdata.min()} --- took {time.time() - t0} seconds")



if DEBUG_NO_VAMPNET:
    interface=None
else:
    interface = load_interface(model_choice="opera")

load_state.loaded = True

def main():
    if PROFILE:
        import yappi
        yappi.start()

    try:
        audio_init()
        st = State()
        st.playing = True

        from functools import partial
        cb = partial(callback, st)

        with term.fullscreen(), term.cbreak():
            with sd.Stream(channels=num_channels, callback=cb, blocksize=blocksize, prime_output_buffers_using_stream_callback=True, dtype=np.int16):
                while True:
                    with term.hidden_cursor():
                        if DEBUG: 
                            time.sleep(100)
                        else:
                            draw_looper(st)

    except KeyboardInterrupt:
        print(term.clear)
        if PROFILE: 
            yappi.stop()

            # retrieve thread stats by their thread id (given by yappi)
            threads = yappi.get_thread_stats()
            for thread in threads:
                print(
                    "Function stats for (%s) (%d)" % (thread.name, thread.id)
                )  # it is the Thread.__class__.__name__
                yappi.get_func_stats(ctx_id=thread.id).print_all()

main()