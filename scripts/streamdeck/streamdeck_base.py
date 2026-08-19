#!/usr/bin/env python3

import os
import json
import threading
import subprocess
import signal
import shutil
from datetime import datetime
from rclpy.node import Node
from PIL import Image, ImageDraw, ImageFont
from StreamDeck.DeviceManager import DeviceManager
from StreamDeck.ImageHelpers import PILHelper

class StreamDeckBase(Node):
    def __init__(self, node_name, config_filename, config):
        super().__init__(node_name)

        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.assets_path = os.path.join(os.path.dirname(self.base_path), '/assets')
        self.config_path = os.path.join(self.assets_path, config_filename)
        self.font_path = os.path.join(self.assets_path, 'Roboto-Regular.ttf')

        self.deck_lock = threading.Lock()

        # --- Bag Recording Setup ---
        self.demo_name = config.get("demo_name", "default_demo")
        self.bag_base_dir = config.get("bag_base_dir", "/datasets")
        self.record_topics = config.get("topics", [])

        self.recording_process = None
        self.current_bag_dir = None
        self.record_active = False
        self.recording_key_index = None
        self.motion_flash_timer = None
        self.flash_state = False

        # Core Colors for the 3 allowed buttons
        self.colors = {
            "inactive":     "#E2F0CB",
            "active":       "#C7CEEA",
            "recording":    "#FF9999",
            "cancel":       "#FFB7B2",
            "home":         "#A0E7E5",
            "text":         "#333333",
            "border":       "#F9F9F9",
            "error":        "#FF0000"
        }

        if not os.path.exists(self.config_path):
            self.get_logger().error(f"Config not found at {self.config_path}")
            return

        with open(self.config_path, "r") as f:
            self.button_layout = json.load(f)

        try:
            self.deck = DeviceManager().enumerate()[0]
            self.deck.open()
            print(self.deck.device.device_info)
            self.deck.reset()
            self.deck.set_brightness(100)
            self.rows, self.cols = self.deck.key_layout()
            self.deck.set_key_callback(self.on_key_change)
        except IndexError:
            self.get_logger().error("No Stream Deck found.")
            return

        self.font = ImageFont.truetype(self.font_path, 14) if os.path.exists(self.font_path) else None
        self.buttons = {}
        print("got here")
        # Initialize the core buttons universally
        self.init_buttons()
        self.render_all_buttons()

    def _get_key_index(self, col, row):
        return (row * self.cols) + col

    def init_buttons(self):
        """Maps the 3 core buttons. Child classes can extend this if needed."""
        for label, coords in self.button_layout.items():
            key_index = self._get_key_index(coords[0], coords[1])
            bg_color = self.colors["inactive"]
            display_label = label
            callback = self.press_default

            if label == "RECORD":
                self.recording_key_index = key_index
                callback = self.press_record
            elif label == "CANCEL_RECORDING":
                bg_color = self.colors["cancel"]
                display_label = "CANCEL\nRECORD"
                callback = self.press_cancel_recording
            elif label == "HOME":
                bg_color = self.colors["home"]
                callback = self.press_home

            self.buttons[key_index] = {
                "label": display_label,
                "callback": callback,
                "bg_color": bg_color,
                "text_color": self.colors["text"]
            }

    # --- Core Universal Actions ---

    def press_record(self, key_index):
        """Toggle recording state to collect rosbag data."""
        self.record_active = not self.record_active

        if self.record_active:
            # START RECORDING
            save_dir = os.path.join(self.bag_base_dir, self.demo_name)
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = f"{self.demo_name}_{timestamp}"
            self.current_bag_dir = os.path.join(save_dir, bag_name)

            cmd = ['ros2', 'bag', 'record', '-o', self.current_bag_dir] + self.record_topics
            self.recording_process = subprocess.Popen(cmd)

            self.get_logger().info(f"Recording Started: {self.current_bag_dir}")
            if not self.motion_flash_timer:
                self.motion_flash_timer = self.create_timer(0.5, self._flash_motion_buttons)
        else:
            # STOP RECORDING (SAVE)
            if self.recording_process is not None:
                self.recording_process.send_signal(signal.SIGINT)
                self.recording_process.wait()
                self.recording_process = None

            self.get_logger().info("Recording Stopped and Saved.")
            if self.motion_flash_timer:
                self.motion_flash_timer.cancel()
                self.motion_flash_timer = None
            self.render_all_buttons()

    def _flash_motion_buttons(self):
        self.flash_state = not self.flash_state
        if self.record_active and self.recording_key_index is not None:
            bg = self.colors["recording"] if self.flash_state else self.colors["inactive"]
            self.update_button_visual(self.recording_key_index, "RECORDING", bg, self.colors["text"])

    def press_cancel_recording(self, key_index):
        self.get_logger().info("Canceling Recording...")

        # STOP AND DELETE
        if self.recording_process is not None:
            self.recording_process.send_signal(signal.SIGINT)
            self.recording_process.wait()
            self.recording_process = None

        if self.current_bag_dir and os.path.exists(self.current_bag_dir):
            shutil.rmtree(self.current_bag_dir)
            self.get_logger().info(f"Deleted bag folder: {self.current_bag_dir}")

        self.record_active = False

        if self.motion_flash_timer:
            self.motion_flash_timer.cancel()
            self.motion_flash_timer = None

        self.render_all_buttons()
        self.press_default(key_index)

    def press_home(self, key_index):
        """To be overridden by robot-specific classes."""
        self.get_logger().warn("Home button pressed, but homing logic is not implemented in the base class.")
        self.press_default(key_index)

    # --- UI & Hardware Visuals ---

    def render_all_buttons(self):
        for key_index, config in self.buttons.items():
            self.update_button_visual(key_index, config["label"], config["bg_color"], config["text_color"])

    def update_button_visual(self, key_index, label, bg_color, text_color):
        if not hasattr(self, 'deck') or not self.deck: return

        img = Image.new("RGB", (100, 100), bg_color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (99, 99)], outline=self.colors["border"], width=5)

        if self.font:
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=self.font)
                w, h = right - left, bottom - top
            except AttributeError:
                w, h = draw.textsize(label, font=self.font)
            draw.text(((100-w)/2, (100-h)/2), label, font=self.font, fill=text_color)

        native_img = PILHelper.to_native_format(self.deck, img)

        with self.deck_lock:
            try:
                self.deck.set_key_image(key_index, native_img)
            except Exception as e:
                self.get_logger().error(f"Deck image transfer error: {e}")

    def on_key_change(self, deck, key, state):
        if state and key in self.buttons:
            self.buttons[key]["callback"](key)

    def _schedule_one_shot(self, duration, callback_func):
        timer_ref = [None]
        def one_shot_wrapper():
            callback_func()
            if timer_ref[0]:
                timer_ref[0].cancel()
                self.destroy_timer(timer_ref[0])
        timer_ref[0] = self.create_timer(duration, one_shot_wrapper)

    def press_default(self, key_index):
        c = self.buttons[key_index]
        self.update_button_visual(key_index, c["label"], self.colors["active"], self.colors["text"])
        self._schedule_one_shot(0.3, lambda: self.update_button_visual(key_index, c["label"], c["bg_color"], c["text_color"]))

    def shutdown(self):
        if self.recording_process is not None:
            self.recording_process.send_signal(signal.SIGINT)
            self.recording_process.wait()

        if hasattr(self, 'deck') and self.deck:
            self.deck.reset()
            self.deck.close()