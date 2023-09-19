from pathlib import Path
import argbind
import audiotools as at
from audiotools import AudioSignal
from multiprocessing import Pool, cpu_count, Manager, Process
from queue import Empty
from tqdm import tqdm

from vampnet.condition import REGISTRY, ConditionFeatures, ChromaStemConditioner, YamnetConditioner

ChromaStemConditioner = argbind.bind(ChromaStemConditioner)
YamnetConditioner = argbind.bind(YamnetConditioner)
conditioner = None

def load_audio(kwargs):
    audio_file = kwargs['audio_file']
    audio_folder = kwargs['audio_folder']
    in_queue = kwargs['queue']
    try:
        sig = AudioSignal(audio_file)
        in_queue.put({'audio_file': audio_file, 'signal': sig})
    except Exception as e:
        print(f"failed to load {audio_file.name} with error {e}")
        print(f"skipping {audio_file.name}")

def process_audio(kwargs, pbar):
    conditioner = kwargs['conditioner']
    in_queue = kwargs['in_queue']
    out_queue = kwargs['out_queue']
    audio_folder = kwargs['audio_folder']
    output_folder = kwargs['output_folder']

    while True:
        try:
            item = in_queue.get()
            if item == 'done':
                break

            audio_file = item['audio_file']
            sig = item['signal']

            output_path = Path(output_folder) / audio_file.with_suffix(".emb")
            if output_path.exists():
                print(f"Output for {audio_file.name} already exists. Skipping processing.")
                continue

            features = ConditionFeatures(
                audio_path=str(audio_file.relative_to(audio_folder)),
                features=conditioner.condition(sig).cpu().numpy(),
                metadata={}
            )
            out_queue.put({'audio_file': audio_file, 'features': features})

            # Update progress bar
            pbar.set_description(f"In Queue: {in_queue.qsize()} | Out Queue: {out_queue.qsize()}")
            pbar.update(1)
        except Empty:
            break

def save_audio_file(kwargs):
    out_queue = kwargs['queue']
    output_folder = kwargs['output_folder']
    while True:
        try:
            item = out_queue.get()
            if item == 'done':
                break
            audio_file = item['audio_file']
            features = item['features']
            output_path = Path(output_folder) / audio_file.with_suffix(".emb")
            if not output_path.exists():
                output_path.parent.mkdir(exist_ok=True, parents=True)
                features.save(output_path)
        except Empty:
            break

@argbind.bind(without_prefix=True)
def condition_and_save(
    audio_folder: str = None,
    output_folder: str= None,
    conditioner_name: str = "yamnet",
    processes: int = cpu_count()
):
    assert audio_folder is not None, "audio_folder must be specified"
    assert output_folder is not None, "output_folder must be specified"

    audio_files = list(at.util.find_audio(Path(audio_folder)))

    conditioner = REGISTRY[conditioner_name]()

    manager = Manager()
    audio_queue = manager.Queue(maxsize=processes*8)
    out_queue = manager.Queue(maxsize=processes*8)

    # Initialize progress bar
    pbar = tqdm(total=len(audio_files), desc="Processing files", dynamic_ncols=True)

    # Start audio loading processes
    args = [{'audio_file': audio_file, 'audio_folder': audio_folder, 'queue': audio_queue} for audio_file in audio_files]
    pool = Pool(processes=processes)
    pool.map_async(load_audio, args)

    # Start saving process
    save_audio_args = {
        'queue': out_queue,
        'output_folder': output_folder
    }
    save_audio_proc = Process(target=save_audio_file, args=(save_audio_args,))
    save_audio_proc.start()

    # Start audio processing process (main process)
    process_audio_args = {
        'conditioner': conditioner,
        'in_queue': audio_queue,
        'out_queue': out_queue,
        'audio_folder': audio_folder,
        'output_folder': output_folder
    }
    process_audio(process_audio_args, pbar)

    pool.close()
    pool.join()

    # Sending the "done" status
    audio_queue.put('done')
    out_queue.put('done')

    save_audio_proc.join()
    pbar.close()

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()
