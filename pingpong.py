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
TAG_DETECT_WIDTH = 960      # detect on a downscaled copy: 1080p is too slow

# --- hitting ---------------------------------------------------------------
HIT_RADIUS = 0.09           # paddle reach, as a fraction of frame width
EARLY_S = 0.25              # a hit may land this long before the ball arrives...
LATE_S = 0.15               # ...or this long after
SWING_WINDOW_S = 0.25       # the swing must peak within this long of contact
PADDLE_REACH = 0.35         # the hub sits past the wrist: wrist + 0.35*(wrist-elbow)

# Swing threshold on |gyro| (x, y, z combined, roughly deg/s). Press `c` in
# game to measure your own swing and set it from that.
SWING_THRESHOLD = 150.0
CALIBRATE_S = 2.0
CALIBRATE_FRACTION = 0.4    # threshold = this fraction of your calibration peak
# In --sim there is no gyro, so swing speed is the wrist's speed on screen,
# in frame widths per second.
SIM_SWING_THRESHOLD = 1.2

# --- table geometry (fractions of the frame) -------------------------------
VANISH = (0.5, 0.14)        # far end of the table, where serves come from
FAR_DEPTH = 6.0             # ball starts 6x further away than the contact plane
BALL_RADIUS = 0.035         # apparent ball radius at the contact plane
ARC = 0.10                  # height of the ball's arc over the net

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
    """Reports a tag id once it has been held in view for TAG_HOLD_S."""

    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICT),
            cv2.aruco.DetectorParameters())
        self.seen_id = None
        self.seen_since = 0.0
        self.fired = False

    def read(self, frame, now):
        """Return (held tag id or None, list of (id, corners) in frame pixels)."""
        h, w = frame.shape[:2]
        scale = min(1.0, TAG_DETECT_WIDTH / w)
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if i == START_TAG or i in LEVELS:
                    found.append((int(i), c.reshape(4, 2) / scale))

        # Debounce: the same tag held still for TAG_HOLD_S fires exactly once.
        current = found[0][0] if found else None
        if current != self.seen_id:
            self.seen_id, self.seen_since, self.fired = current, now, False
        if current is not None and not self.fired and now - self.seen_since >= TAG_HOLD_S:
            self.fired = True
            return current, found
        return None, found


# --- the ball --------------------------------------------------------------

class Ball:
    """A ball moving between the far end of the table and the contact plane.

    Depth runs from FAR_DEPTH to 1 (incoming) or 1 to FAR_DEPTH (returning),
    linearly in time; apparent size and distance from the vanishing point
    both go as 1/depth, which is what makes it look like it is coming at you.
    """

    def __init__(self, target, now, duration, incoming=True):
        self.target = target          # (x, y) in frame fractions at depth 1
        self.t0 = now
        self.duration = duration
        self.incoming = incoming

    @property
    def arrival(self):
        return self.t0 + self.duration

    def progress(self, now):
        return (now - self.t0) / self.duration

    def screen(self, now, w, h):
        """Return (x, y, radius) in pixels."""
        p = self.progress(now)
        if not self.incoming:
            p = 1.0 - p
        depth = max(0.4, FAR_DEPTH - (FAR_DEPTH - 1.0) * p)
        s_far = 1.0 / FAR_DEPTH
        k = (1.0 / depth - s_far) / (1.0 - s_far)   # 0 at the far end, 1 at contact
        vx, vy = VANISH
        x = vx + (self.target[0] - vx) * k
        y = vy + (self.target[1] - vy) * k
        y -= ARC * math.sin(math.pi * clamp(p, 0, 1)) * k   # arc over the net
        r = BALL_RADIUS * w / depth
        return int(x * w), int(y * h), max(2, int(r))


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


def read_pose(landmarks, hand):
    """Return (paddle point, torso) in frame fractions; either may be None.

    Torso is (centre x, shoulder height, shoulder width) and is what serves
    aim at, so the ball always comes somewhere the player can reach.
    """
    wrist = landmarks[LM.RIGHT_WRIST if hand == "right" else LM.LEFT_WRIST]
    elbow = landmarks[LM.RIGHT_ELBOW if hand == "right" else LM.LEFT_ELBOW]
    paddle = None
    if visible(wrist):
        px, py = wrist.x, wrist.y
        if visible(elbow):
            px += PADDLE_REACH * (wrist.x - elbow.x)
            py += PADDLE_REACH * (wrist.y - elbow.y)
        paddle = (px, py)

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


# --- drawing ---------------------------------------------------------------

def draw_table(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    vx, vy = VANISH
    far_half = 0.12
    pts = np.array([[(vx - far_half) * w, vy * h], [(vx + far_half) * w, vy * h],
                    [0.98 * w, 0.98 * h], [0.02 * w, 0.98 * h]], np.int32)
    cv2.fillPoly(overlay, [pts], (90, 50, 10))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    cv2.line(frame, (int(vx * w), int(vy * h)), (int(vx * w), int(0.98 * h)),
             (255, 255, 255), 1)
    # The net, about a third of the way down in screen space.
    k = 0.28
    ny = int((vy + (0.98 - vy) * k) * h)
    nx0 = int(((vx - far_half) + ((0.02) - (vx - far_half)) * k) * w)
    nx1 = int(((vx + far_half) + ((0.98) - (vx + far_half)) * k) * w)
    cv2.line(frame, (nx0, ny), (nx1, ny), (230, 230, 230), 3)


def text(frame, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def centred(frame, s, y, scale=1.4, color=(255, 255, 255)):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, 3)
    text(frame, s, ((frame.shape[1] - tw) // 2, y), scale, color, 3)


def draw_hud(frame, game, paddle_xy, swing, now, sim, tags):
    h, w = frame.shape[:2]

    for tag_id, corners in tags:
        pts = corners.astype(np.int32)
        # Tags are detected on the raw frame; mirror to match the display.
        pts[:, 0] = w - 1 - pts[:, 0]
        cv2.polylines(frame, [pts], True, (0, 220, 255), 3)
        label = "START" if tag_id == START_TAG else f"LEVEL {tag_id}"
        text(frame, label, tuple(pts.min(axis=0) - [0, 8]), 0.7, (0, 220, 255))

    if game.ball is not None and game.state == PLAYING:
        bx, by, br = game.ball.screen(now, w, h)
        shadow_y = int(min(0.97, game.ball.target[1] + 0.04) * h) if game.ball.incoming else by
        cv2.ellipse(frame, (bx, max(by, shadow_y)), (br, max(1, br // 3)), 0, 0, 360,
                    (40, 40, 40), -1)
        cv2.circle(frame, (bx, by), br, (0, 165, 255), -1)
        cv2.circle(frame, (bx, by), br, (255, 255, 255), 1)

    if paddle_xy is not None:
        px, py = int(paddle_xy[0] * w), int(paddle_xy[1] * h)
        swinging = swing.peak(now, SWING_WINDOW_S) >= game.swing_threshold
        colour = (0, 255, 0) if swinging else (0, 0, 255)
        cv2.circle(frame, (px, py), int(HIT_RADIUS * w), colour, 2)
        cv2.circle(frame, (px, py), 6, colour, -1)

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

    cap, start_ms = pick_camera()
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

            frame = cv2.flip(raw, 1)
            h, w = frame.shape[:2]
            ts_ms = int(time.time() * 1000) - start_ms
            result = landmarker.detect_for_video(
                Image(image_format=ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)

            paddle_xy = None
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)
                paddle_xy, seen_torso = read_pose(landmarks, hand)
                if seen_torso is not None:
                    torso = seen_torso

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
            draw_hud(frame, game, paddle_xy, swing, now, args.sim, seen)
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
