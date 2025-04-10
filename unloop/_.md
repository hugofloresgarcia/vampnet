## client side setup
clone
```
https://github.com/hugofloresgarcia/unsound-objects.git
git checkout unloop
```

install
```
conda create -n unsound python=3.10
conda activate unsound
pip install -r requirements.txt
```

## server side setup
ssh into malleus
```
ssh bryan@malleus.cs.northwestern.edu -L 7860:localhost:7860
```

then leave the malleus window open and start up a new local window

(kindly ask hugo to launch the gradio on port 7860)

you can verify that the gradio is running by opening `http://localhost:7860` on your browser

## launch the gradio server (vampnet)
you have to run the gradio server running vampnet model.
(on the remote machine)
```bash
conda create -n vampnet python=3.10
git clone https://github.com/huggingface.co/spaces/hugggof/vampnet-music.git
pip install -e .
CUDA_VISIBLE_DEVICES=0 python app.py
```

### launch the gradio server (s2s)
you have to run the gradio server running audit model.

(on the remote machine)
```bash
conda create -n audit python=3.10
cd audit
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python scripts/text2sfx/demo.py ckpts/adobe-soda/checkpoints/seethara/text2sfx/25-02-18-256ch-8s/ --model latest_ema.pth
```

or for audit-old
```
CUDA_VISIBLE_DEVICES=0 python scripts/cdit/demos/voice2sfx.py ckpts/rms-centroid-ppg/latest.pth
```

## launch the client (laptop)
then launch the client from your local terminal
```
python client.py --vampnet_url <VAMPNET_URL> --s2s_url http://localhost:7860
```

## max setup
Then...make sure you have installed (in Max)
```
flucoma
```

MAKE SURE YOU ARE RUNNING MAX 8.  It is not compatible with Max 9.

Now open up the right max patch `./max/sound-objects.maxpat`. 

### text prompts
NOTE: text prompts are from the list here
https://universalcategorysystem.com/
https://www.dropbox.com/scl/fo/lw1i20cgsm4edsvj3awn1/AP_ZhzG3LlpfFLbX309FbOU?dl=0&e=1&preview=UCS+v8.2.1+Full+List.xlsx&rlkey=wa2onzo0difpew1nze6odztlp
*** HUGO make an empty 'audio' directory in the repo! ***
