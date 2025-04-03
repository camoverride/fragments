# Swarm

Average together a display faces [[link](https://smith.cam/swarm.html)]


## Setup

This is intended to be run on a pi5.

- `git clone git@github.com:camoverride/fragments.git`
- `cd fragments`
- `git checkout swarm`
- `python -m venv --system-site-packages .venv`
- `source .venv/bin/activate`

Install this package for installing dlib:
- `pip install setuptools`

Install cmake which is requied by dlib which is in turn required by face_recognition:
- `sudo apt update`
- `sudo apt install cmake`
- `sudo apt install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev`
- `pip install dlib -vvv`

Install remaining requirements:
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`


## Test

- `export DISPLAY=:0`
- `python run_display.py`


## Configure

Adjust image margins and dimensions, face recognition tolerance, blur threshold, etc. in `config.yaml`.


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies. This is expected to run on a Raspberry Pi 5:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u display.service`
