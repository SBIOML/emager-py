import tkinter as tk
from tkinter import ttk
import sv_ttk
from PIL import ImageTk, Image
import os
import json
from libemg.screen_guided_training import ScreenGuidedTraining


def get_gestures_from_libemg(gestures: list, out_path: str, img_format="png"):
    train_ui = ScreenGuidedTraining()
    out_path += "/"
    # Download gestures with indices 1,2,3,4,5 and store them in the "gestures/" folder
    train_ui.download_gestures(gestures, out_path, download_gifs=True)
    list_file = list(filter(lambda f: f.endswith("json"), os.listdir(out_path)))[0]
    gestures_name = []
    with open(out_path + list_file, "r") as f:
        gestures_dict = json.load(f)
        for g in gestures:
            gestures_name.append(out_path + gestures_dict[str(g)] + "." + img_format)
    return gestures_name


class EmagerGuidedTraining:
    def __init__(
        self,
        reps: int = 5,
        gestures: list[int] = [2, 14, 26, 1, 14, 30],
        gestures_path: str = "gestures",
        resume_training_callback: callable = None,
    ):
        self.root = tk.Tk()
        self.root.title("EMaGer Guided Training")
        self.root.resizable(True, True)
        self.root.attributes("-type", "dialog")

        sv_ttk.set_theme("dark")

        self.state = 0  # idle
        self.total_reps = reps
        self.current_rep = 0
        self.image = None
        self.gestures = get_gestures_from_libemg(gestures, gestures_path)
        self.gesture_index = 0
        self.resume_training_callback = resume_training_callback

        self.create_widgets()

        self.root.bind("<Escape>", lambda e: self.cancel_training())
        self.root.bind("<Return>", lambda e: self.continue_training())

    def countdown(self, remaining, callback=None):
        # change text in label
        if remaining > 0:
            self.timer["text"] = f"{remaining:.1f} seconds remaining"
            self.root.after(100, self.countdown, remaining - 0.1, callback)
        elif callback is not None:
            self.timer["text"] = ""

            callback()

    def create_widgets(self):
        self.info_lbl = ttk.Label(
            self.root,
            text="Press Continue to start.",
        )
        self.timer = ttk.Label(self.root)
        self.continue_btn = ttk.Button(
            self.root, text="Continue", command=self.continue_training
        )
        self.cancel_btn = ttk.Button(
            self.root, text="Cancel", command=self.cancel_training
        )
        self.image_lbl = tk.Label(self.root)
        self.set_picture(True)

        self.info_lbl.grid(row=0, column=0, columnspan=2)
        self.timer.grid(row=1, column=0, columnspan=2)
        self.continue_btn.grid(row=2, column=0, sticky="nsew")
        self.cancel_btn.grid(row=2, column=1, sticky="nsew")
        self.image_lbl.grid(row=3, column=0, columnspan=2, sticky="nsew")

    def continue_training(self):
        if self.current_rep >= self.total_reps:
            self.info_lbl["text"] = "Training completed, exiting soon..."
            self.continue_btn["state"] = tk.DISABLED
            self.countdown(5, self.cancel_training)
            return

        if self.state == 0:
            self.state = 1
            self.continue_btn["state"] = tk.DISABLED
            self.info_lbl["text"] = (
                f"Rep {self.current_rep+1}/{self.total_reps} Gesture {self.gesture_index+1}/{len(self.gestures)}"
            )
            self.set_picture(True)
            self.countdown(2, self.continue_training)
        elif self.state == 1:
            self.state = 2
            self.set_picture(False)
            if self.resume_training_callback is not None:
                self.resume_training_callback()
            self.countdown(5, self.continue_training)
        elif self.state == 2:
            self.state = 0
            self.gesture_index += 1
            if self.gesture_index >= len(self.gestures):
                # Done with this rep
                self.current_rep += 1
                self.gesture_index = 0
            self.continue_btn["state"] = tk.ACTIVE
            self.timer["text"] = "Press Continue"

    def cancel_training(self):
        self.root.quit()

    def set_picture(self, grayscale=False):
        if grayscale:
            self.image = (
                Image.open(self.gestures[self.gesture_index])
                .convert("L")
                .resize((400, 400))
            )
        else:
            self.image = Image.open(self.gestures[self.gesture_index]).resize(
                (400, 400)
            )
        image = ImageTk.PhotoImage(self.image)
        self.image_lbl.configure(image=image)
        self.image_lbl.image = image

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmagerGuidedTraining()
    app.start()
