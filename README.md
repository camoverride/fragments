# Fragments 🧩

An automated portrait that collects the faces of visitors, combines them together, and
dissolves them into fragments [[link](https://smith.cam/fragments.html)]


## Setup

This is intended to be run on a Raspberry Pi 5 with a pi camera.

- `git clone git@github.com:camoverride/fragments.git`
- `cd fragments`
- `python3 -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`


## Test

- `export DISPLAY=:0`
- `python run_display.py`


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
