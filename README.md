# Fragments 🧩

An automated portrait that collects the faces of visitors, combines them together, and
dissolves them into fragments [[link](https://smith.cam/fragments.html)]


## Setup

This is intended to be run on a Raspberry Pi 5 with a pi camera.

- `git clone git@github.com:camoverride/fragments.git`
- `cd fragments`
- `python3 -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
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

Install unclutter:
- `sudo apt-get install unclutter`

Create the databases and clear existing files:
- `python _database_utils.py`


## Test

- `export DISPLAY=:0`
- `python run_display.py`
- If you change the image parameters or want a fresh start: `python _database_utils.py`


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

## TODO

- [X] add config
- [X] code cleanup
- [ ] integrate picam
- [ ] fix issue with simple_crop getting too-large area
- [ ] install on Rpi
- [ ] test on Rpi
- [X] integrate swapping between Fragments and Averages
- [ ] currently, collages look bad when using random phone data, check to make sure they look good in the wild too!

- [ ] add randomness in sampling (recency bias)
- [ ] improve logging
- [ ] test threading, locks, etc. Fails if things dont write, uneven list lens for in-mem mode etc

- [ ] explicitly set `swarm`, `fragments` or `both` mode