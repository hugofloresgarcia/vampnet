#!/bin/bash
set -e

if [ -z "${USER}" ]; then
  echo "We need USER to be set!"; exit 100
fi

# check if host uid and gid are set
if [ -z "${HOST_USER_ID}" ]; then
    echo "Please set HOST_USER_ID env. variables to continue." ; exit 0
fi

if [ -z "${HOST_USER_GID}" ]; then
    echo "Please set HOST_USER_GID env. variables to continue." ; exit 0
fi

USER_ID=$HOST_USER_ID
USER_GID=$HOST_USER_GID
USER_HOME=/u/home

# modify uid and gid to match host
sed -i -e "s/^${USER}:\([^:]*\):[0-9]*:[0-9]*/${USER}:\1:${USER_ID}:${USER_GID}/"  /etc/passwd

# create a group for host gid
groupadd -f --gid "${USER_GID}" "host_group"

chown $USER_ID $USER_HOME
chown $USER_ID /u/home/.zshrc
chown $USER_ID /u/home/.oh-my-zsh

mkdir -p /u/home/.cache
chown -R $USER_ID:$USER_GID /u/home/.cache/

_term() {
  echo "Caught SIGTERM signal!"
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM

su -p "${USER}"
