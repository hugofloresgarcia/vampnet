FROM us.gcr.io/lyrebird-research/research-image/audio:beta

COPY requirements.txt requirements.txt
ARG GITHUB_TOKEN
RUN echo machine github.com login ${GITHUB_TOKEN} > ~/.netrc

COPY env/alias.sh /alias.sh
COPY env/entry_script.sh /entry_script.sh
RUN cat /alias.sh >> ~/.zshrc

# USER researcher
RUN pip install --upgrade -r requirements.txt
RUN pip install --upgrade tensorflow
RUN pip install --upgrade librosa
RUN pip install --upgrade numba
RUN pip install protobuf==3.20
ENV PYTHONPATH "$PYTHONPATH:/u/home/src"
ENV NUMBA_CACHE_DIR=/tmp/

USER root
RUN wget https://github.com/jgm/pandoc/releases/download/2.18/pandoc-2.18-1-amd64.deb
RUN dpkg -i pandoc-2.18-1-amd64.deb
RUN apt-get update && apt-get install task-spooler

RUN head -n -1 /entry_script.sh > /entry_script_jupyter.sh
RUN head -n -1 /entry_script.sh > /entry_script_tensorboard.sh
RUN head -n -1 /entry_script.sh > /entry_script_gradio.sh

RUN echo \
    'su -p ${USER} -c "source ~/.zshrc && jupyter lab --ip=0.0.0.0"' >> \
    /entry_script_jupyter.sh
RUN echo \
    'su -p ${USER} -c "source ~/.zshrc && tensorboard --logdir=$TENSORBOARD_PATH --samples_per_plugin audio=500 --bind_all"' >> \
    /entry_script_tensorboard.sh
RUN echo \
    'su -p ${USER} -c "source ~/.zshrc && python app.py --args.load=conf/app.yml"' >> \
    /entry_script_gradio.sh
