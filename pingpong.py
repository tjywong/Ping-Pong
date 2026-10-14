"""Virtual ping-pong with a LEGO Education Double Motor as the paddle.

You hold the Double Motor hub like a paddle and stand in front of the webcam.
A virtual ball flies at you down an on-screen table. To return it, two things
have to be true at the moment it arrives:

  - Pose (MediaPipe): your paddle hand is where the ball is on screen.
  - IMU (the hub's gyroscope): you are actually swinging, not just holding
    your hand in the right spot.

AprilTags (tag36h11) run the game from across the room, no keyboard needed:

  - tag 1..5   pick the player level, which sets the ball speed
  - tag 0      start the game (a short countdown, then the first serve)

The record number of continuous hits is published live over MQTT to
ME193/Rogers/<name> as a floating point number, e.g. "12.0".

    source .venv/bin/activate
    python pingpong.py                        # connect to the hub and play
    python pingpong.py --sim                  # no hub: wrist speed stands in for the gyro
    python pingpong.py --card-color orange --card-serial 1129
    python pingpong.py --pick-camera          # choose a camera instead of the built-in one

Keys while the game window has focus:
    space   start (same as tag 0)        1-5   pick level (same as tags 1-5)
    c       calibrate the swing threshold (swing hard for 2 s)
    h       switch paddle hand           r     reset the record
    q/ESC   quit
"""

import argparse
import collections
import json
import math
import os
import random
import subprocess
import time
import urllib.request

import cv2
import numpy as np

import legoeducation as le
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from camlib import pick_camera
from lelib import doubleMotor
from mqttlib import MQTTClient

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
RECORD_PATH = os.path.join(HERE, "record.json")

# --- AprilTags -------------------------------------------------------------
TAG_DICT = cv2.aruco.DICT_APRILTAG_36H11
START_TAG = 0
# Player level -> seconds the ball takes to travel the length of the table.
LEVELS = {1: 2.0, 2: 1.6, 3: 1.3, 4: 1.05, 5: 0.85}
TAG_HOLD_S = 0.4            # a tag must stay in view this long to count
TAG_DROPOUT_S = 0.35        # ...but may drop out of detection this long without resetting
TAG_DETECT_WIDTH = 1280     # only downscale frames wider than this (1080p is too slow)

# --- hitting ---------------------------------------------------------------
HIT_RADIUS = 0.09           # paddle reach (blade + ball), as a fraction of frame width
EARLY_S = 0.25              # a hit may land this long before the ball arrives...
LATE_S = 0.15               # ...or this long after
SWING_WINDOW_S = 0.25       # the swing must peak within this long of contact

# The paddle is drawn in the hand: the handle runs from the wrist out along the
# hand, and the blade sits on the end of it. Sizes are fractions of frame width.
HANDLE_LEN = 0.045
HANDLE_WIDTH = 0.018
BLADE_RADIUS = 0.06

# Swing threshold on |gyro| (x, y, z combined, roughly deg/s). Press `c` in
# game to measure your own swing and set it from that.
SWING_THRESHOLD = 150.0
CALIBRATE_S = 2.0
CALIBRATE_FRACTION = 0.4    # threshold = this fraction of your calibration peak
# In --sim there is no gyro, so swing speed is the wrist's speed on screen,
# in frame widths per second.
SIM_SWING_THRESHOLD = 1.2

# --- table geometry --------------------------------------------------------
# A real pinhole projection, so the table lies flat on its plane. World units
# are metres: x to the right, y up from the table top, z away from the screen.
FOCAL = 0.9                 # focal length, in frame widths
HORIZON = 0.42              # horizon height on screen, as a fraction of frame height
EYE_HEIGHT = 0.50           # camera height above the table top: lower = flatter table
TABLE_NEAR = 1.7            # distance from the screen to the near edge of the table
TABLE_LENGTH = 2.74         # regulation table: 2.74 m x 1.525 m, net 15.25 cm
TABLE_HALF_WIDTH = 0.7625
TABLE_THICKNESS = 0.04
NET_HEIGHT = 0.1525
TABLE_FAR = TABLE_NEAR + TABLE_LENGTH
NET_Z = TABLE_NEAR + TABLE_LENGTH / 2
CONTACT_Z = 1.0             # where you hit the ball: just off the near edge
BALL_SIZE = 0.035           # ball radius (drawn bigger than a real 20 mm ball)
TABLE_ALPHA = 0.6           # how solid the table is over the camera image (0..1)

COUNTDOWN_S = 3.0
GAME_OVER_S = 2.5
BUZZ_MS = 90                # haptic buzz on a hit
BUZZ_SPEED = 70

FONT = cv2.FONT_HERSHEY_SIMPLEX
LM = mp.solutions.pose.PoseLandmark


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_record():
    try:
        with open(RECORD_PATH) as f:
            return int(json.load(f)["record"])
    except (OSError, ValueError, KeyError):
        return 0


def save_record(record):
    with open(RECORD_PATH, "w") as f:
        json.dump({"record": record}, f)


# --- hardware --------------------------------------------------------------

class Paddle:
    """The Double Motor hub, held in the hand. Reads the gyro, buzzes on hits."""

    def __init__(self, enabled, card_color=None, card_serial=None, buzz=True):
        self.enabled = enabled
        self.card_color = card_color
        self.card_serial = card_serial
        self.buzz_enabled = buzz
        self.hub = None

    def connect(self):
        if not self.enabled:
            print("[paddle] --sim: no hub, wrist speed stands in for the gyro")
            return
        print("[paddle] searching for the Double Motor over Bluetooth...")
        self.hub = doubleMotor()
        self.hub.connect(card_serial=self.card_serial, card_color=self.card_color)
        print("[paddle] connected")

    def gyro_magnitude(self):
        """|gyro| across all three axes, or 0 before the first IMU report."""
        imu = self.hub.imu_device
        g = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
        if any(math.isnan(v) for v in g):
            return 0.0
        return math.sqrt(sum(v * v for v in g))

    def buzz(self, ms=BUZZ_MS):
        if self.hub is None or not self.buzz_enabled:
            return
        try:
            self.hub.motor_run_for_time(ms, motor=le.MOTOR_BOTH, speed=BUZZ_SPEED,
                                        blocking=False)
        except Exception as e:  # a dropped buzz must never end the rally
            print(f"[paddle] buzz failed: {e}")

    def close(self):
        if self.hub is not None:
            try:
                self.hub.motor_stop(motor=le.MOTOR_BOTH)
                self.hub.disconnect()
            except Exception:
                pass


class Scoreboard:
    """Publishes the record streak to ME193/Rogers/<name> as a float."""

    def __init__(self, name, broker, enabled=True):
        self.topic = f"ME193/Rogers/{name}"
        self.broker = broker
        self.client = None
        if not enabled:
            return
        try:
            self.client = MQTTClient(broker=broker)
            self.client.connect()
            print(f"[mqtt] publishing record to {broker} topic {self.topic}")
        except Exception as e:
            self.client = None
            print(f"[mqtt] could not reach {broker} ({e}); playing offline")

    @property
    def online(self):
        return self.client is not None

    def publish(self, record):
        if self.client is not None:
            # Retained, so anyone who subscribes later still sees the record.
            self.client.publish(self.topic, str(float(record)), retain=True)

    def close(self):
        if self.client is not None:
            self.client.disconnect()


class SwingMeter:
    """Rolling history of swing speed, so a hit can look back at the peak."""

    def __init__(self, keep_s=2.0):
        self.keep_s = keep_s
        self.samples = collections.deque()

    def add(self, t, value):
        self.samples.append((t, value))
        while self.samples and self.samples[0][0] < t - self.keep_s:
            self.samples.popleft()

    def peak(self, now, window):
        return max((v for t, v in self.samples if t >= now - window), default=0.0)

    def latest(self):
        return self.samples[-1][1] if self.samples else 0.0


class TagReader:
    """Reports a tag id once it has been held in view for TAG_HOLD_S.

    Detection flickers on a hand-held tag (motion blur, glare, a tilt), so a
    tag only loses its place in the hold if it goes undetected for longer
    than TAG_DROPOUT_S, not on the first missed frame.
    """

    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICT),
            cv2.aruco.DetectorParameters())
        self.first_seen = {}          # tag id -> when this hold began
        self.last_seen = {}           # tag id -> last frame it was detected
        self.fired = set()            # ids that already fired during this hold

    def read(self, frame, now):
        """Return (held tag id or None, [(id, corners in frame pixels, hold 0..1)])."""
        h, w = frame.shape[:2]
        scale = min(1.0, TAG_DETECT_WIDTH / w)
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                i = int(i)
                if i == START_TAG or i in LEVELS:
                    detected.append((i, c.reshape(4, 2) / scale))

        for i, _ in detected:
            if i not in self.last_seen:
                self.first_seen[i] = now
                print(f"[tags] saw tag {i}")
            self.last_seen[i] = now
        for i in [i for i, t in self.last_seen.items() if now - t > TAG_DROPOUT_S]:
            del self.last_seen[i], self.first_seen[i]   # gone: showing it again fires again
            self.fired.discard(i)

        held = None
        for i in self.last_seen:
            if i not in self.fired and now - self.first_seen[i] >= TAG_HOLD_S:
                self.fired.add(i)
                held = i
                break
        found = [(i, c, clamp((now - self.first_seen[i]) / TAG_HOLD_S, 0, 1))
                 for i, c in detected]
        return held, found


# --- the ball --------------------------------------------------------------

def project(x, y, z, w, h):
    """World point (metres) -> screen pixels."""
    f = FOCAL * w
    z = max(z, 0.2)
    return w / 2 + f * x / z, HORIZON * h + f * (EYE_HEIGHT - y) / z


def unproject(px, py, z, w, h):
    """Screen pixels at depth z -> world point."""
    f = FOCAL * w
    return (px - w / 2) * z / f, EYE_HEIGHT - (py - HORIZON * h) * z / f, z


class Ball:
    """A ball in flight over the table, in two legs.

    Incoming: opponent -> bounce on your side -> the contact point.
    Outgoing: contact point -> bounce on their side -> opponent.
    Each leg is a straight line in x and z with a parabolic arc in height,
    and the ball moves at a constant speed along z.
    """

    def __init__(self, target, now, duration, incoming=True):
        self.target = target          # contact point on screen, in frame fractions
        self.t0 = now
        self.duration = duration
        self.incoming = incoming

    @property
    def arrival(self):
        return self.t0 + self.duration

    def _legs(self, w, h):
        contact = unproject(self.target[0] * w, self.target[1] * h, CONTACT_Z, w, h)
        cx = contact[0]
        opponent = (cx * 0.4, 0.30, TABLE_FAR + 0.4)
        if self.incoming:
            bounce = (cx * 0.8, 0.0, TABLE_NEAR + 0.25 * TABLE_LENGTH)
            return [(opponent, bounce, 0.18), (bounce, contact, 0.10)]
        bounce = (cx * 0.5, 0.0, TABLE_NEAR + 0.78 * TABLE_LENGTH)
        return [(contact, bounce, 0.20), (bounce, opponent, 0.08)]

    def position(self, now, w, h):
        """World position; keeps going along the last leg after arrival."""
        legs = self._legs(w, h)
        lengths = [abs(b[2] - a[2]) for a, b, _ in legs]
        d = (now - self.t0) / self.duration * sum(lengths)
        for n, ((a, b, arc), length) in enumerate(zip(legs, lengths)):
            if d <= length or n == len(legs) - 1:
                s = max(0.0, d / length)
                break
            d -= length
        x = a[0] + (b[0] - a[0]) * s
        y = a[1] + (b[1] - a[1]) * s + arc * 4 * s * (1 - s)
        z = a[2] + (b[2] - a[2]) * s
        return x, y, z

    def screen(self, now, w, h):
        """Return (x, y, radius) in pixels."""
        x, y, z = self.position(now, w, h)
        px, py = project(x, y, z, w, h)
        r = FOCAL * w * BALL_SIZE / max(z, 0.2)
        return int(px), int(py), max(2, int(r))

    def shadow(self, now, w, h):
        """(x, y, radius) of the ball's shadow on the table, or None if off it."""
        x, y, z = self.position(now, w, h)
        if y < 0 or not TABLE_NEAR <= z <= TABLE_FAR or abs(x) > TABLE_HALF_WIDTH:
            return None
        px, py = project(x, 0.0, z, w, h)
        return int(px), int(py), max(2, int(FOCAL * w * BALL_SIZE / z))

    def depth(self, now, w, h):
        return self.position(now, w, h)[2]


# --- the game --------------------------------------------------------------

WAITING, COUNTDOWN, PLAYING, GAME_OVER, CALIBRATING = (
    "WAITING", "COUNTDOWN", "PLAYING", "GAME OVER", "CALIBRATING")


class Game:
    def __init__(self, paddle, scoreboard, swing_threshold):
        self.paddle = paddle
        self.scoreboard = scoreboard
        self.swing_threshold = swing_threshold
        self.state = WAITING
        self.state_since = time.monotonic()
        self.level = 1
        self.streak = 0
        self.record = load_record()
        self.ball = None
        self.flash = None             # (text, colour, until)
        self.calib_peak = 0.0
        self.scoreboard.publish(self.record)

    def set_state(self, state, now):
        self.state, self.state_since = state, now

    def set_level(self, level, now):
        self.level = level
        self.flash = (f"LEVEL {level}", (0, 220, 255), now + 1.2)
        print(f"[game] level {level} ({LEVELS[level]:.2f} s per flight)")

    def start(self, now):
        if self.state in (WAITING, GAME_OVER):
            self.streak = 0
            self.ball = None
            self.set_state(COUNTDOWN, now)
            print(f"[game] starting at level {self.level}")

    def start_calibration(self, now):
        if self.state in (WAITING, GAME_OVER):
            self.calib_peak = 0.0
            self.set_state(CALIBRATING, now)

    def reset_record(self, now):
        self.record = 0
        save_record(0)
        self.scoreboard.publish(0)
        self.flash = ("RECORD RESET", (200, 200, 200), now + 1.2)

    def serve(self, now, torso):
        """Serve a new ball at a random spot the player can reach."""
        cx, cy, span = torso
        tx = clamp(cx + random.uniform(-1.4, 1.4) * span, 0.12, 0.88)
        ty = clamp(cy + random.uniform(-0.2, 1.3) * span, 0.30, 0.90)
        self.ball = Ball((tx, ty), now, LEVELS[self.level], incoming=True)
        self.reached = False          # was the paddle ever on the ball this window?

    def on_hit(self, now, contact):
        self.streak += 1
        self.paddle.buzz()
        if self.streak > self.record:
            self.record = self.streak
            save_record(self.record)
        self.scoreboard.publish(self.record)
        # Send it back the way it came; the opponent returns it on arrival.
        self.ball = Ball(contact, now, LEVELS[self.level] * 0.8, incoming=False)
        self.flash = ("HIT!", (0, 255, 0), now + 0.4)

    def on_miss(self, now, why):
        print(f"[game] miss ({why}) -- streak {self.streak}, record {self.record}")
        self.paddle.buzz(350)
        self.scoreboard.publish(self.record)
        self.flash = (f"MISS: {why}", (0, 0, 255), now + GAME_OVER_S)
        self.set_state(GAME_OVER, now)

    def update(self, now, paddle_xy, swing, torso, w, h):
        """Advance one frame. paddle_xy is in frame fractions, or None."""
        if self.state == CALIBRATING:
            self.calib_peak = max(self.calib_peak, swing.latest())
            if now - self.state_since >= CALIBRATE_S:
                if self.calib_peak > 0:
                    self.swing_threshold = CALIBRATE_FRACTION * self.calib_peak
                    print(f"[game] swing peak {self.calib_peak:.1f} -> "
                          f"threshold {self.swing_threshold:.1f}")
                self.set_state(WAITING, now)
            return

        if self.state == COUNTDOWN and now - self.state_since >= COUNTDOWN_S:
            self.set_state(PLAYING, now)
            self.serve(now, torso)
            return

        if self.state == GAME_OVER and now - self.state_since >= GAME_OVER_S:
            self.set_state(WAITING, now)
            return

        if self.state != PLAYING or self.ball is None:
            return

        ball = self.ball
        if not ball.incoming:
            if now >= ball.arrival:
                self.serve(now, torso)   # the opponent returns it
            return

        # Incoming: inside the hit window, need both reach (pose) and swing (IMU).
        if now < ball.arrival - EARLY_S:
            return
        if now > ball.arrival + LATE_S:
            if self.reached:
                why = "no swing"
            elif paddle_xy is None:
                why = "can't see you"
            else:
                why = "too far"
            self.on_miss(now, why)
            return
        if paddle_xy is not None and self._near(paddle_xy, ball, now, w, h):
            self.reached = True
            if swing.peak(now, SWING_WINDOW_S) >= self.swing_threshold:
                self.on_hit(now, ball.target)

    @staticmethod
    def _near(paddle_xy, ball, now, w, h):
        bx, by, _ = ball.screen(now, w, h)
        dx = paddle_xy[0] * w - bx
        dy = paddle_xy[1] * h - by
        return math.hypot(dx, dy) <= HIT_RADIUS * w


# --- pose ------------------------------------------------------------------

def visible(lm, threshold=0.5):
    return getattr(lm, "visibility", 1.0) >= threshold


class Point:
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


def mirror(landmarks):
    """Flip landmarks left-right to match the mirrored display.

    Pose runs on the raw camera frame, where the model's RIGHT_WRIST really is
    the player's right wrist. Running it on the mirrored frame instead would
    swap the hands, because a mirrored person looks like one facing the other way.
    """
    return [Point(1.0 - lm.x, lm.y, getattr(lm, "visibility", 1.0)) for lm in landmarks]


def read_pose(landmarks, hand, w, h):
    """Return (paddle, torso); either may be None.

    The paddle is a dict: "center" is the blade centre in frame fractions (what
    the ball has to touch), "wrist" is in pixels, and "dir" is the unit
    direction the paddle points, wrist to blade, in pixels.

    Torso is (centre x, shoulder height, shoulder width) and is what serves
    aim at, so the ball always comes somewhere the player can reach.
    """
    side = "RIGHT" if hand == "right" else "LEFT"
    wrist = landmarks[LM[f"{side}_WRIST"]]
    elbow = landmarks[LM[f"{side}_ELBOW"]]
    knuckles = [landmarks[LM[f"{side}_INDEX"]], landmarks[LM[f"{side}_PINKY"]]]
    paddle = None
    if visible(wrist):
        wx, wy = wrist.x * w, wrist.y * h
        # Point along the hand if we can see it, otherwise along the forearm.
        if all(visible(k, 0.3) for k in knuckles):
            dx = sum(k.x for k in knuckles) / 2 * w - wx
            dy = sum(k.y for k in knuckles) / 2 * h - wy
        elif visible(elbow):
            dx, dy = wx - elbow.x * w, wy - elbow.y * h
        else:
            dx, dy = 0.0, -1.0           # straight up
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        reach = (HANDLE_LEN + BLADE_RADIUS) * w
        paddle = {"center": ((wx + dx * reach) / w, (wy + dy * reach) / h),
                  "wrist": (wx, wy), "dir": (dx, dy)}

    torso = None
    ls, rs = landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]
    if visible(ls) and visible(rs):
        span = max(0.08, abs(ls.x - rs.x))
        torso = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2, span)
    return paddle, torso


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        a, b = landmarks[c.start], landmarks[c.end]
        cv2.line(frame, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)),
                 (90, 160, 90), 1)


def draw_paddle(frame, paddle, swinging):
    """A ping-pong paddle in the hand: wooden handle, red rubber blade."""
    w = frame.shape[1]
    wx, wy = paddle["wrist"]
    dx, dy = paddle["dir"]
    nx, ny = -dy, dx                       # across the handle
    hw = HANDLE_WIDTH * w / 2
    blade_r = BLADE_RADIUS * w
    start = (wx - dx * 0.25 * HANDLE_LEN * w, wy - dy * 0.25 * HANDLE_LEN * w)
    end = (wx + dx * (HANDLE_LEN * w + 0.4 * blade_r), wy + dy * (HANDLE_LEN * w + 0.4 * blade_r))
    handle = np.array([[start[0] + nx * hw, start[1] + ny * hw],
                       [end[0] + nx * hw, end[1] + ny * hw],
                       [end[0] - nx * hw, end[1] - ny * hw],
                       [start[0] - nx * hw, start[1] - ny * hw]], np.int32)
    cv2.fillPoly(frame, [handle], (70, 130, 190), cv2.LINE_AA)      # wood
    cv2.polylines(frame, [handle], True, (30, 60, 100), 2, cv2.LINE_AA)
    # Grain: a darker stripe down the middle of the handle.
    cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
             (50, 100, 160), 1, cv2.LINE_AA)

    cx, cy = (int(paddle["center"][0] * w), int(paddle["center"][1] * frame.shape[0]))
    r = int(blade_r)
    if swinging:                                                     # swing glow
        cv2.circle(frame, (cx, cy), r + 7, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (25, 25, 25), -1, cv2.LINE_AA)    # wood edge
    cv2.circle(frame, (cx, cy), r - 4, (40, 40, 200), -1, cv2.LINE_AA)  # red rubber
    cv2.circle(frame, (cx, cy), int(r * 0.78), (30, 30, 170), 1, cv2.LINE_AA)
    hl = (int(cx - dx * r * 0.35 + nx * r * 0.35), int(cy - dy * r * 0.35 + ny * r * 0.35))
    cv2.circle(frame, hl, max(2, r // 6), (110, 110, 255), -1, cv2.LINE_AA)  # sheen


# --- drawing ---------------------------------------------------------------

def _pts(*world, w, h):
    return np.array([project(*p, w, h) for p in world], np.int32)


def draw_table(frame):
    """The table top and its front edge, blended over the camera image."""
    h, w = frame.shape[:2]
    hw, t = TABLE_HALF_WIDTH, TABLE_THICKNESS
    top = _pts((-hw, 0, TABLE_NEAR), (hw, 0, TABLE_NEAR), (hw, 0, TABLE_FAR),
               (-hw, 0, TABLE_FAR), w=w, h=h)
    front = _pts((-hw, 0, TABLE_NEAR), (hw, 0, TABLE_NEAR), (hw, -t, TABLE_NEAR),
                 (-hw, -t, TABLE_NEAR), w=w, h=h)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [top], (110, 60, 15), cv2.LINE_AA)
    cv2.fillPoly(overlay, [front], (60, 30, 5), cv2.LINE_AA)
    cv2.addWeighted(overlay, TABLE_ALPHA, frame, 1 - TABLE_ALPHA, 0, frame)
    cv2.polylines(frame, [top], True, (255, 255, 255), 3, cv2.LINE_AA)
    centre = _pts((0, 0, TABLE_NEAR), (0, 0, TABLE_FAR), w=w, h=h)
    cv2.line(frame, tuple(centre[0]), tuple(centre[1]), (255, 255, 255), 2, cv2.LINE_AA)


def draw_net(frame):
    h, w = frame.shape[:2]
    hw = TABLE_HALF_WIDTH + 0.15
    net = _pts((-hw, 0, NET_Z), (hw, 0, NET_Z), (hw, NET_HEIGHT, NET_Z),
               (-hw, NET_HEIGHT, NET_Z), w=w, h=h)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [net], (200, 200, 200), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.line(frame, tuple(net[3]), tuple(net[2]), (250, 250, 250), 3, cv2.LINE_AA)  # tape
    for a, b in ((net[0], net[3]), (net[1], net[2])):                                # posts
        cv2.line(frame, tuple(a), tuple(b), (40, 40, 40), 4, cv2.LINE_AA)


def draw_ball(frame, ball, now):
    h, w = frame.shape[:2]
    shadow = ball.shadow(now, w, h)
    if shadow is not None:
        sx, sy, sr = shadow
        overlay = frame.copy()
        cv2.ellipse(overlay, (sx, sy), (sr, max(1, sr // 3)), 0, 0, 360, (20, 20, 20), -1,
                    cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    bx, by, br = ball.screen(now, w, h)
    cv2.circle(frame, (bx, by), br, (0, 150, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, by), br, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (bx - br // 3, by - br // 3), max(1, br // 4), (170, 220, 255), -1,
               cv2.LINE_AA)


def text(frame, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def centred(frame, s, y, scale=1.4, color=(255, 255, 255)):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, 3)
    text(frame, s, ((frame.shape[1] - tw) // 2, y), scale, color, 3)


def draw_hud(frame, game, paddle, swing, now, sim, tags):
    h, w = frame.shape[:2]

    for tag_id, corners, hold in tags:
        pts = corners.astype(np.int32)
        # Tags are detected on the raw frame; mirror to match the display.
        pts[:, 0] = w - 1 - pts[:, 0]
        colour = (0, 255, 0) if hold >= 1 else (0, 220, 255)
        cv2.polylines(frame, [pts], True, colour, 3, cv2.LINE_AA)
        label = "START" if tag_id == START_TAG else f"LEVEL {tag_id}"
        x0, y0 = pts.min(axis=0)
        text(frame, label, (int(x0), int(y0) - 12), 0.7, colour)
        # Hold bar: fills while the tag is held, fires when full.
        bw = int(pts[:, 0].max() - x0)
        cv2.rectangle(frame, (int(x0), int(y0) - 8), (int(x0) + bw, int(y0) - 3), (60, 60, 60), -1)
        cv2.rectangle(frame, (int(x0), int(y0) - 8), (int(x0) + int(bw * hold), int(y0) - 3),
                      colour, -1)

    # The paddle goes under the ball, so you can see the ball meet it.
    if paddle is not None:
        draw_paddle(frame, paddle, swing.peak(now, SWING_WINDOW_S) >= game.swing_threshold)

    # The net hides a ball that is still on the far side of it.
    ball = game.ball if game.state == PLAYING else None
    if ball is not None and ball.depth(now, w, h) > NET_Z:
        draw_ball(frame, ball, now)
        draw_net(frame)
    else:
        draw_net(frame)
        if ball is not None:
            draw_ball(frame, ball, now)


    # Score panel.
    text(frame, f"Streak {game.streak}", (14, 34), 0.9, (255, 255, 255))
    text(frame, f"Record {game.record}", (14, 66), 0.9, (0, 255, 255))
    text(frame, f"Level {game.level}  ({LEVELS[game.level]:.2f}s)", (14, 96), 0.6)
    mqtt = f"MQTT {game.scoreboard.topic}" if game.scoreboard.online else "MQTT offline"
    text(frame, mqtt, (14, 122), 0.5,
         (0, 200, 0) if game.scoreboard.online else (0, 120, 255), 1)
    text(frame, "SIM (wrist speed)" if sim else "HUB gyro", (14, 144), 0.5,
         (0, 165, 255) if sim else (0, 200, 0), 1)

    # Swing meter: live value against the threshold.
    mx, my, mw, mh = 14, h - 40, 260, 18
    top = max(game.swing_threshold * 2.5, 1e-6)
    fill = int(mw * clamp(swing.latest() / top, 0, 1))
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (60, 60, 60), -1)
    cv2.rectangle(frame, (mx, my), (mx + fill, my + mh), (0, 200, 255), -1)
    tx = mx + int(mw / 2.5)
    cv2.line(frame, (tx, my - 4), (tx, my + mh + 4), (255, 255, 255), 2)
    unit = "w/s" if sim else "dps"
    text(frame, f"swing {swing.latest():.1f} / {game.swing_threshold:.1f} {unit}",
         (mx, my - 8), 0.5, (255, 255, 255), 1)

    # State banner.
    if game.state == WAITING:
        centred(frame, "Show tag 1-5 for level, tag 0 to start", int(h * 0.55), 0.9)
    elif game.state == COUNTDOWN:
        left = COUNTDOWN_S - (now - game.state_since)
        centred(frame, str(max(1, math.ceil(left))), int(h * 0.55), 3.0, (0, 255, 255))
    elif game.state == CALIBRATING:
        centred(frame, f"SWING HARD!  peak {game.calib_peak:.1f}", int(h * 0.55), 1.1,
                (0, 255, 255))
    if game.flash and now < game.flash[2]:
        centred(frame, game.flash[0], int(h * 0.42), 1.6, game.flash[1])


# --- camera ----------------------------------------------------------------

def builtin_camera_index():
    """OpenCV index of the Mac's built-in camera, or None if there isn't one.

    OpenCV's AVFoundation backend numbers cameras in order of their unique id,
    not the order System Settings lists them -- with an iPhone attached over
    Continuity Camera the iPhone is 0 and the built-in camera is 1.
    """
    try:
        raw = subprocess.check_output(["system_profiler", "SPCameraDataType", "-json"],
                                      text=True, stderr=subprocess.DEVNULL)
        cams = json.loads(raw).get("SPCameraDataType", [])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    cams.sort(key=lambda c: c.get("spcamera_unique-id", ""))
    for i, c in enumerate(cams):
        name = f"{c.get('_name', '')} {c.get('spcamera_model-id', '')}"
        if "FaceTime" in name or "Built-in" in name:
            return i
    return None


def open_camera(index, width=1280, height=720):
    """Open a camera by index; returns (cap, start_ms) like camlib.pick_camera."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. "
                           "Check System Settings -> Privacy & Security -> Camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.5)                  # let the sensor settle
    return cap, int(time.time() * 1000)


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Virtual ping-pong with a LEGO Double Motor paddle.")
    ap.add_argument("--sim", action="store_true", help="no hub: use wrist speed as the swing signal")
    ap.add_argument("--name", default="Tyler", help="MQTT topic is ME193/Rogers/<name>")
    ap.add_argument("--broker", default="test.mosquitto.org")
    ap.add_argument("--no-mqtt", action="store_true", help="don't publish the score")
    ap.add_argument("--card-color", default=None,
                    help="Connection Card colour, e.g. orange (default: first hub found)")
    ap.add_argument("--card-serial", default=None, help="Connection Card serial, e.g. 1129")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="paddle hand")
    ap.add_argument("--swing-threshold", type=float, default=None,
                    help=f"|gyro| that counts as a swing (default {SWING_THRESHOLD})")
    ap.add_argument("--no-buzz", action="store_true", help="don't buzz the motors on a hit")
    ap.add_argument("--camera", type=int, default=None,
                    help="OpenCV camera index (default: the Mac's built-in camera)")
    ap.add_argument("--pick-camera", action="store_true",
                    help="choose the camera from live previews instead")
    args = ap.parse_args()

    card_color = None
    if args.card_color:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}")

    ensure_model()
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH,
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))

    paddle = Paddle(not args.sim, card_color, args.card_serial, buzz=not args.no_buzz)
    paddle.connect()
    scoreboard = Scoreboard(args.name, args.broker, enabled=not args.no_mqtt)

    threshold = args.swing_threshold
    if threshold is None:
        threshold = SIM_SWING_THRESHOLD if args.sim else SWING_THRESHOLD
    game = Game(paddle, scoreboard, threshold)
    tags = TagReader()
    swing = SwingMeter()

    camera = args.camera
    if camera is None and not args.pick_camera:
        camera = builtin_camera_index()
        if camera is None:
            print("[camera] no built-in camera found; pick one")
    if camera is None:
        cap, start_ms = pick_camera()
    else:
        print(f"[camera] using camera {camera}")
        cap, start_ms = open_camera(camera)
    print(f"[camera] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    hand = args.hand
    last_paddle = None
    torso = (0.5, 0.45, 0.2)        # sensible default until we first see shoulders
    bad_reads = 0
    print("Show tag 1-5 to pick a level, tag 0 (or space) to start. q to quit.")

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                bad_reads += 1
                if bad_reads > 150:
                    raise RuntimeError("camera stopped delivering frames")
                continue
            bad_reads = 0
            now = time.monotonic()

            # AprilTags are read off the raw frame: mirrored, they don't decode.
            seen = []
            if game.state in (WAITING, GAME_OVER):
                held, seen = tags.read(raw, now)
                if held == START_TAG:
                    game.start(now)
                elif held in LEVELS:
                    game.set_level(held, now)

            # Pose runs on the raw frame too, then gets mirrored (see mirror()).
            ts_ms = int(time.time() * 1000) - start_ms
            result = landmarker.detect_for_video(
                Image(image_format=ImageFormat.SRGB,
                      data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)), ts_ms)
            frame = cv2.flip(raw, 1)
            h, w = frame.shape[:2]

            held = None             # where the paddle is drawn this frame
            if result.pose_landmarks:
                landmarks = mirror(result.pose_landmarks[0])
                draw_skeleton(frame, landmarks)
                held, seen_torso = read_pose(landmarks, hand, w, h)
                if seen_torso is not None:
                    torso = seen_torso
            paddle_xy = held["center"] if held is not None else None

            # Swing signal: the hub's gyro, or in --sim the paddle's screen speed.
            if args.sim:
                speed = 0.0
                if paddle_xy is not None and last_paddle is not None:
                    (x0, y0, t0), (x1, y1) = last_paddle, paddle_xy
                    if now > t0:
                        speed = math.hypot(x1 - x0, (y1 - y0) * h / w) / (now - t0)
                swing.add(now, speed)
                last_paddle = (*paddle_xy, now) if paddle_xy is not None else None
            else:
                swing.add(now, paddle.gyro_magnitude())

            game.update(now, paddle_xy, swing, torso, w, h)

            draw_table(frame)
            draw_hud(frame, game, held, swing, now, args.sim, seen)
            cv2.imshow("Ping Pong", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                game.start(now)
            elif ord("1") <= key <= ord("5") and game.state in (WAITING, GAME_OVER):
                game.set_level(key - ord("0"), now)
            elif key == ord("c"):
                game.start_calibration(now)
            elif key == ord("h"):
                hand = "left" if hand == "right" else "right"
                print(f"[game] paddle hand: {hand}")
            elif key == ord("r"):
                game.reset_record(now)
    except KeyboardInterrupt:
        pass
    finally:
        paddle.close()
        scoreboard.close()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
