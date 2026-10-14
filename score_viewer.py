"""Test window for the ping-pong score: shows what arrives over MQTT.

Subscribes to the topics pingpong.py publishes to and displays each message
as it lands: the live streak (ME193/Rogers/<name>) big, the record
(ME193/Rogers/<name>/record) under it, and a log of recent messages with the
time they arrived. Both are published retained, so the current values show up
as soon as this connects, even mid-game.

    source .venv/bin/activate
    python score_viewer.py                    # ME193/Rogers/Tyler
    python score_viewer.py --name Alex        # someone else's score
    python score_viewer.py --all              # everyone under ME193/Rogers/#

Press q or ESC in the window to quit.
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np

from mqttlib import MQTTClient

W, H = 720, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
LOG_LINES = 10


def text(img, s, org, scale=0.6, color=(230, 230, 230), thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Show ping-pong scores arriving over MQTT.")
    ap.add_argument("--name", default="Tyler", help="watch ME193/Rogers/<name>")
    ap.add_argument("--all", action="store_true", help="watch every topic under ME193/Rogers/")
    ap.add_argument("--broker", default="test.mosquitto.org")
    args = ap.parse_args()

    topic = "ME193/Rogers/#" if args.all else f"ME193/Rogers/{args.name}"
    lock = threading.Lock()
    log = collections.deque(maxlen=LOG_LINES)     # (arrival time, topic, payload)
    latest = {}                                   # topic -> (payload, arrival time)

    def on_message(t, payload):
        # Runs on the MQTT network thread; the window reads under the lock.
        with lock:
            now = time.time()
            log.appendleft((now, t, payload))
            latest[t] = (payload, now)

    client = MQTTClient(broker=args.broker)
    client.connect()
    if not client._client.is_connected():
        raise SystemExit(f"could not connect to {args.broker}")
    # mqttlib routes callbacks by exact topic string, which never matches a
    # wildcard; hook paho directly so `#` works too.
    client._client.on_message = lambda c, u, msg: on_message(msg.topic, msg.payload.decode())
    if args.all:
        client._client.subscribe(topic)
    else:
        client._client.subscribe([(topic, 0), (f"{topic}/record", 0)])
    print(f"[viewer] {args.broker}: listening on {topic}")

    try:
        while True:
            img = np.full((H, W, 3), (30, 26, 22), np.uint8)
            text(img, "MQTT score viewer", (20, 36), 0.8, (255, 255, 255), 2)
            text(img, f"{args.broker}  /  {topic}", (20, 64), 0.5, (0, 200, 255))
            with lock:
                entries = sorted(latest.items(), key=lambda kv: -kv[1][1])
                rows = list(log)

            if not entries:
                text(img, "waiting for a message...", (20, 150), 0.9, (150, 150, 150), 2)
            elif not args.all:
                with lock:
                    live = latest.get(topic)
                    best = latest.get(f"{topic}/record")
                if live is not None:
                    payload, at = live
                    fresh = time.time() - at < 1.0      # flash green when it changes
                    text(img, payload, (20, 180), 3.0, (0, 255, 0) if fresh else (0, 255, 255), 6)
                    text(img, f"live streak   (updated "
                              f"{time.strftime('%H:%M:%S', time.localtime(at))})",
                         (24, 208), 0.55, (180, 180, 180))
                if best is not None:
                    text(img, f"record  {best[0]}", (24, 238), 0.7, (255, 255, 255), 2)
            else:
                # Leaderboard: highest score first; anything that isn't a number last.
                def score(kv):
                    try:
                        return -float(kv[1][0])
                    except ValueError:
                        return float("inf")
                # One row per player: their live topic, not the /record ones.
                players = [kv for kv in entries if kv[0].count("/") == 2]
                for n, (t, (payload, at)) in enumerate(sorted(players, key=score)[:5]):
                    y = 110 + n * 34
                    fresh = time.time() - at < 1.0
                    text(img, t.removeprefix("ME193/Rogers/"), (24, y), 0.75)
                    text(img, payload, (300, y), 0.75, (0, 255, 0) if fresh else (0, 255, 255), 2)

            cv2.line(img, (20, 256), (W - 20, 256), (80, 80, 80), 1)
            text(img, "recent messages", (20, 278), 0.5, (150, 150, 150))
            for n, (at, t, payload) in enumerate(rows):
                y = 300 + n * 18
                if y > H - 8:
                    break
                stamp = time.strftime("%H:%M:%S", time.localtime(at))
                text(img, f"{stamp}  {t}  {payload}", (24, y), 0.45)

            cv2.imshow("MQTT score viewer", img)
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
