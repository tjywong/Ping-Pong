# Ping Pong

Virtual ping-pong with a LEGO Education Double Motor as the paddle. Hold the
hub in your hand, stand in front of the webcam, and return the ball as it
flies at you down the on-screen table.

| Input | What it does |
|---|---|
| **Pose** (MediaPipe, webcam) | Places the paddle: your wrist, extended a little past it along the forearm to where the hub sits. The ball only counts as reached if the paddle circle is on it. |
| **IMU** (the hub's gyroscope) | Detects whether you are actually swinging. Holding still in the right spot is a miss. |
| **AprilTags** (tag36h11) | Tags **1–5** pick the player level (ball speed). Tag **0** starts the game. |
| **MQTT** | The record number of continuous hits is published live to `ME193/Rogers/Tyler` on `test.mosquitto.org` as a float (e.g. `12.0`), retained. |

A hit needs both at once: the paddle within reach of the ball **and** a swing
peak above the threshold, inside a window from 0.25 s before the ball arrives
to 0.15 s after. On a miss the game tells you why (`too far`, `no swing`,
`can't see you`) and returns to the waiting screen.

## Levels

| Tag | Level | Seconds per flight |
|---|---|---|
| 1 | 1 | 2.00 |
| 2 | 2 | 1.60 |
| 3 | 3 | 1.30 |
| 4 | 4 | 1.05 |
| 5 | 5 | 0.85 |

Printable tags are in [`tags/`](tags/). Hold one up to the camera for about
half a second. It needs a white border around it to be detected.

## Setup

```bash
cd ~/Ping\ Pong
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pose model (`pose_landmarker_lite.task`) downloads itself on the first run.

## Running

```bash
source .venv/bin/activate

python pingpong.py --sim          # no hub: wrist speed on screen stands in for the gyro
python pingpong.py                # connect to the first Double Motor found
python pingpong.py --card-color orange --card-serial 1129   # a specific hub
python pingpong.py --name Tyler   # MQTT topic ME193/Rogers/<name>
```

When `pingpong.py` starts, it asks which camera to use. Then:

1. **Calibrate** (recommended the first time): press `c` and swing hard for
   2 seconds. The swing threshold is set to 40% of your peak. The meter at
   the bottom-left shows the live swing value against the threshold, and the
   paddle circle turns green while you're swinging.
2. Show a level tag (1–5), then tag 0. After a 3-second countdown the first ball is served.

Keys: `space` start · `1`–`5` level · `c` calibrate swing · `h` switch
paddle hand · `r` reset record · `q`/`Esc` quit.

Other flags: `--hand left`, `--swing-threshold N`, `--no-buzz` (the motors
buzz briefly on a hit), `--no-mqtt`, `--broker HOST`.

Watch the score from another terminal:

```bash
mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -v
```

The record is also saved to `record.json`, so it survives restarts.

## Files

- `pingpong.py`: the game
- `lelib.py`, `mqttlib.py`, `camlib.py`: ME193 class libraries (hub, MQTT, camera picker)
- `tags/`: printable AprilTags 0–5

## Notes

- `mediapipe` is pinned to 0.10.21 because 1.x crashes on Apple Silicon. It
  brings in `opencv-contrib-python` 4.11, which has `cv2.aruco`. Don't also
  install `opencv-python`.
- AprilTags are detected on the raw camera frame. The display is mirrored, and a
  mirrored tag doesn't decode.
- The first run triggers macOS's camera permission prompt for your terminal.
