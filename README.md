# Ping Pong

Virtual ping-pong with a LEGO Education Double Motor as the paddle. Hold the
hub in your hand, stand in front of the webcam, and return the ball as it
flies at you down the on-screen table.

| Input | What it does |
|---|---|
| **Pose** (MediaPipe, webcam) | Draws a ping-pong paddle in your right hand (handle at the wrist, pointing along your hand). The ball only counts as reached if it touches the paddle's blade. |
| **IMU** (the hub's gyroscope) | Detects whether you are actually swinging. Holding still in the right spot is a miss, and so is shaking the hub. |
| **AprilTags** (tag36h11) | Tags **1–5** pick the player level (ball speed). Tag **0** starts the game. |
| **MQTT** | Your live streak of continuous hits is published to `ME193/Rogers/Tyler` on `test.mosquitto.org` as a float (`1.0`, `2.0`, …, back to `0.0` on a miss). Your record goes to `ME193/Rogers/Tyler/record`. Both are retained. |

A hit needs both at once: the paddle within reach of the ball **and** a real
swing, inside a window from 0.25 s before the ball arrives to 0.15 s after.

A real swing is **one stroke**. The hub has to keep turning the same way,
without reversing, through at least 50° over at least 0.08 s, while your
wrist moves at least 6% of the screen width. "The same way" is judged on the
hub's whole rotation, not axis by axis, so a diagonal swing counts in full.
Shaking the hub fails on both counts: it reverses direction many times a
second, so no single turn adds up to much, and your wrist hardly moves on
screen.

You swing *through* the ball. The paddle usually crosses the ball partway
into the stroke, before the stroke has turned the full 50°. So a hit counts
if the paddle was on the ball at any point during a qualifying stroke, give
or take 0.1 s, even if the paddle has moved past it by the time the stroke
qualifies. The gyro is read about 100 times a
second on its own thread, so a fast shake can't be mistaken for one long
turn. In `--sim` there's no gyro, so a swing is your wrist moving one way at
least 15% of the screen width in 0.35 s. Waggling it back and forth doesn't
count. On a miss the game tells you why (`too far`, `no swing`,
`can't see you`) and returns to the waiting screen.

## Levels

| Tag | Level | Seconds per flight |
|---|---|---|
| 1 | 1 | 2.00 |
| 2 | 2 | 1.60 |
| 3 | 3 | 1.30 |
| 4 | 4 | 1.05 |
| 5 | 5 | 0.85 |

Printable tags are in [`tags/`](tags/). Hold one up to the camera until the
bar above it fills (about half a second). It needs a white border around it
to be detected. Each tag the camera sees is logged as `[tags] saw tag N`.

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

It uses the Mac's built-in FaceTime camera. Add `--pick-camera` to choose
from live previews instead, or `--camera N` to give an OpenCV index directly.
(With an iPhone attached over Continuity Camera, OpenCV numbers the iPhone 0
and the built-in camera 1. The game looks the built-in one up by name, so you
don't need to know this.) Then:

1. **Calibrate** (optional): press `c` and take 2–3 real swings in 3
   seconds. The required angle is set to half your biggest swing. The meter
   at the bottom-left shows your biggest recent stroke against the threshold,
   and whether your arm moved. It turns green, and the paddle glows green,
   when a stroke counts. After a "no swing" miss, the log shows what your
   best stroke measured.
2. Show a level tag (1–5), then tag 0. After a 3-second countdown the first ball is served.

Keys: `space` start · `1`–`5` level · `c` calibrate swing · `h` switch
paddle hand · `r` reset record · `q`/`Esc` quit.

Other flags: `--hand left`, `--swing-threshold N`, `--no-buzz`, `--no-mqtt`,
`--broker HOST`.

**Haptics:** the hub has no vibration motor, so the game buzzes its drive
motors. A hit is a short, sharp jolt: three quick full-speed back-and-forth
wiggles, then a hard brake. A miss is a longer rumble. It runs on its own
thread so the game doesn't stall. `HIT_BUZZ` and `MISS_BUZZ` in
`pingpong.py` set the patterns. Haptics are much stronger with something on
the motor shafts, such as a wheel or an off-centre beam.

## Watching the score

`score_viewer.py` opens a window that shows what arrives over MQTT: your live
streak in large type (it flashes green when it changes), your record under
it, and a log of recent messages. Both are published retained, so they show
up the moment the viewer connects. Run it alongside the game, or on another laptop:

```bash
python score_viewer.py                # ME193/Rogers/Tyler
python score_viewer.py --name Alex    # someone else
python score_viewer.py --all          # everyone's live score under ME193/Rogers/
```

The record is also saved to `record.json`, so it survives restarts.

## Files

- `pingpong.py`: the game
- `score_viewer.py`: test window that shows the live score and record arriving over MQTT
- `lelib.py`, `mqttlib.py`, `camlib.py`: ME193 class libraries (hub, MQTT, camera picker)
- `tags/`: printable AprilTags 0–5

## Notes

- `mediapipe` is pinned to 0.10.21 because 1.x crashes on Apple Silicon. It
  brings in `opencv-contrib-python` 4.11, which has `cv2.aruco`. Don't also
  install `opencv-python`.
- AprilTags are detected on the raw camera frame. The display is mirrored, and a
  mirrored tag doesn't decode.
- The table is a regulation 2.74 m × 1.525 m table drawn with a real
  perspective projection. In `pingpong.py`, `EYE_HEIGHT` sets how flat it
  looks (lower is flatter), `TABLE_NEAR` sets how far away it is, and
  `TABLE_ALPHA` sets how solid it is over the camera image (0 = invisible,
  1 = solid).
- The first run triggers macOS's camera permission prompt for your terminal.
