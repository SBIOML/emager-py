import tkinter as tk
from tkinter import PhotoImage, ttk, filedialog
import sv_ttk
from PIL import ImageTk, Image
import os
import csv
import numpy as np
import shutil
import json
import threading
from libemg.screen_guided_training import ScreenGuidedTraining
from emager_py.streamers import EmagerStreamerInterface




def get_all_gestures_from_libemg(out_path: str, img_format="png"):
    train_ui = ScreenGuidedTraining()
    out_path += "/"
    # Download all gestures and store them in the "gestures/" folder
    train_ui.download_gestures(list(range(1,35)), out_path)
    list_file = list(filter(lambda f: f.endswith("json"), os.listdir(out_path)))[0]
    gestures_name = []
    with open(out_path + list_file, "r") as f:
        gestures_dict = json.load(f)
        for g in range(1, 31):
            gestures_name.append(out_path + gestures_dict[str(g)] + "." + img_format)
    return gestures_name

class ImageListbox(tk.Frame):
    def __init__(self, gesture_folder: str = "gesturesALL"):
        self.root = tk.Tk()
        self.root.title("Choose Pictures")
        self.root.resizable(True, True)
        sv_ttk.set_theme("dark")

        self.gesture_folder = gesture_folder
        self.selected_indices = []
        self.create_widgets()

    def create_widgets(self):
        # Create a canvas to hold the images
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the canvas
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold the images
        self.frame = tk.Frame(self.canvas, bg='white')
        self.canvas.create_window((0, 0), window=self.frame, anchor=tk.NW)

        # Bind mousewheel to scroll
        self.root.bind("<MouseWheel>", self.on_mousewheel)

        # Add images to the list
        self.image_paths = ["image1.png", "image2.png", "image3.png"]
        self.image_paths = get_all_gestures_from_libemg(self.gesture_folder)
        self.images = [PhotoImage(file=image_path) for image_path in self.image_paths]

        self.labels = []
        for i, image in enumerate(self.images):
            label = tk.Label(self.frame, image=image, bg='white')
            label.pack(anchor=tk.W)
            label.bind("<Button-1>", lambda event, index=i: self.on_select(index))
            self.labels.append(label)

        # Update the scroll region
        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(-1*(event.delta//120), "units")

    def on_select(self, index):
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            self.labels[index].config(bg='white')
        else:
            self.selected_indices.append(index)
            self.labels[index].config(bg='lightblue')

    def start(self):
        self.root.mainloop()
        return self.image_paths



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
        streamer: EmagerStreamerInterface,
        reps: int = 5,
        training_time: float = 5,
        gestures: list[int] = [2, 14, 26, 1, 14, 30],
        gestures_path: str = "gestures",
        resume_training_callback: callable = None,
        callback_arg: None | str = None,
    ):
        self.root = tk.Tk()
        self.root.title("EMaGer Guided Training")
        self.root.resizable(True, True)

        sv_ttk.set_theme("dark")
        
        self.streamer = streamer
        self.state = 0  # idle
        self.total_reps = reps
        self.current_rep = 0
        self.image = None
        self.gestures_path = gestures_path
        self.gestures = get_gestures_from_libemg(gestures, gestures_path)
        print(self.gestures)
        self.gesture_index = 0
        self.training_time = training_time
        self.resume_training_callback = resume_training_callback
        self.callback_arg = callback_arg
        self.callback_arg_lut = {
            "gesture": self.get_gesture,
        }
        self.callback_thread = threading.Thread()

        self.create_widgets()

        self.root.bind("<Escape>", lambda e: self.cancel_training())
        self.root.bind(
            "<Return>", lambda e: self.continue_training() if self.state == 0 else None
        )
        self.root.protocol("WM_DELETE_WINDOW", self.cancel_training)

    def get_callback_lut(self):
        return self.callback_arg_lut

    def get_gesture(self):
        return self.gesture_index

    def countdown(self, remaining, callback=None):
        # change text in label
        if remaining > 0:
            self.timer["text"] = f"{remaining:.1f} seconds remaining"
            self.root.after(100, self.countdown, remaining - 0.1, callback)
        elif callback is not None:
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

        self.reps_label = ttk.Label(self.root, text="Reps: ")
        self.reps_entry = ttk.Spinbox(
            self.root, from_=1, to=999, 
            command=self.set_reps_and_time, width=3, 
        )
        self.reps_entry.set(str(self.total_reps))
        self.training_time_lbl = ttk.Label(self.root, text="Training Time: ")
        self.training_time_entry = ttk.Spinbox(
            self.root, from_=1, to=999,
            command=self.set_reps_and_time, width=3,
        )
        self.training_time_entry.set(str(self.training_time))
        self.set_reps_and_time()
        
        self.user_id_lbl = ttk.Label(self.root, text="UserID: ", state=tk.DISABLED)
        self.user_id_entry = ttk.Spinbox(
            self.root, from_=0, to=999, 
            command=self.get_folder, width=3,
        )
        self.user_id_entry.set("0")
        self.session_lbl = ttk.Label(self.root, text="Session: ", state=tk.DISABLED)
        self.session_entry = ttk.Spinbox(
            self.root, from_=0, to=999,
            command=self.get_folder, width=3,
        )
        self.session_entry.set("0")
        self.data_dir = os.getcwd()
        self.data_dir_lbl = ttk.Label(self.root, text=self.data_dir)
        self.data_dir_btn = ttk.Button(
            self.root, text="Choose Folder", command=self.choose_folder
        )
        self.get_folder()
        options = ["left", "right"]
        self.arm_lbl = ttk.OptionMenu(self.root, tk.StringVar(), "right", *options)


        self.info_lbl.grid(row=0, column=0, columnspan=5)
        self.timer.grid(row=1, column=0, columnspan=5)
        self.continue_btn.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self.cancel_btn.grid(row=2, column=3, columnspan=2,  sticky="nsew")
        
        self.image_lbl.grid(row=3, column=0, columnspan=5)

        self.reps_label.grid(row=4, column=0, columnspan=1)
        self.reps_entry.grid(row=4, column=1, columnspan=1)
        self.training_time_lbl.grid(row=4, column=2, columnspan=2)
        self.training_time_entry.grid(row=4, column=4, columnspan=1)

        self.user_id_lbl.grid(row=5, column=0, columnspan=1)
        self.user_id_entry.grid(row=5, column=1, columnspan=1)
        self.session_lbl.grid(row=5, column=2, columnspan=1)
        self.session_entry.grid(row=5, column=3, columnspan=1)
        self.arm_lbl.grid(row=5, column=4, columnspan=1)
        self.data_dir_lbl.grid(row=6, column=0, columnspan=4)
        self.data_dir_btn.grid(row=6, column=4, columnspan=1)

    def set_reps_and_time(self):
        self.total_reps = int(self.reps_entry.get())
        self.training_time = float(self.training_time_entry.get())

    def choose_folder(self):
        self.data_dir = filedialog.askdirectory()
        self.get_folder()

    def get_folder(self):
        self.userID = str(self.user_id_entry.get()).zfill(3)
        self.user_dir = os.path.join(self.data_dir, self.userID)
        self.sessionNb = str(self.session_entry.get()).zfill(3)
        self.final_dir = os.path.join(self.user_dir, self.sessionNb)
        self.data_dir_lbl["text"] = self.final_dir
        return self.final_dir

    def save_data(self):
        pass
        data  = np.array(streamer.read())
        data = np.transpose(data)

        self.get_folder()
        self.arm = self.arm_lbl.cget("text")
        self.gestureNb = str(self.gesture_index).zfill(3)
        self.trialNb = str(self.current_rep).zfill(3)

        try:
            os.mkdir(self.user_dir)
        except FileExistsError:
            pass

        try:
            os.mkdir(self.final_dir)
        except FileExistsError:
            pass

        # put data into CSV file
        # aaa-bbb-ccc-ddd.csv --> a = userID, b = session#, c = gesture#, d = trial#
        self.csv_filename = f"{self.userID}-{self.sessionNb}-{self.gestureNb}-{self.trialNb}-{self.arm}.csv"
        self.csv_path = os.path.join(self.final_dir, self.csv_filename)
        print(f"Saving data to {self.csv_path}")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
            f.close()

    def continue_training(self):
        if self.current_rep >= self.total_reps:
            self.info_lbl["text"] = "Training completed, exiting soon..."
            self.continue_btn["state"] = tk.DISABLED
            self.countdown(3, self.cancel_training)
            return

        if self.state == 0:
            # Update gesture info and start preparation countdown
            self.continue_btn["state"] = tk.DISABLED
            self.info_lbl["text"] = (
                f"Rep {self.current_rep+1}/{self.total_reps} Gesture {self.gesture_index+1}/{len(self.gestures)}"
            )
            self.set_picture(True)
            self.countdown(1, self.continue_training)
            self.state = 1
        elif self.state == 1:
            # Start sampling callback
            self.set_picture(False)
            self.save_thread = threading.Thread(
                        target=self.save_data, daemon=True
                    )
            self.save_thread.start()
            if self.resume_training_callback is not None:
                if self.callback_arg is not None:
                    self.callback_thread = threading.Thread(
                        target=self.resume_training_callback,
                        args=(self.callback_arg_lut[self.callback_arg](),),
                        daemon=True,
                    )
                else:
                    self.callback_thread = threading.Thread(
                        target=self.resume_training_callback, daemon=True
                    )
                self.callback_thread.start()
            self.countdown(self.training_time, self.continue_training)
            self.state = 2
        elif self.state == 2:
            # Waiting for thread to finish and wait for continue
            if self.save_thread.is_alive():
                self.timer["text"] = f"Saving data to {self.csv_filename}..."
                self.root.after(150, self.continue_training)
                return
            if self.resume_training_callback is not None:
                if self.callback_thread.is_alive():
                    self.timer["text"] = (
                        f"Waiting for {self.resume_training_callback.__name__} to finish..."
                    )
                    self.root.after(150, self.continue_training)
                    return
            self.gesture_index += 1
            if self.gesture_index >= len(self.gestures):
                # Done with this rep
                self.current_rep += 1
                self.gesture_index = 0
            self.continue_btn["state"] = tk.ACTIVE
            self.timer["text"] = "Press Continue"
            self.state = 0

    def cleanup(self):
        shutil.rmtree(self.gestures_path)

    def cancel_training(self):
        self.root.destroy()

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
    import time
    from emager_py.streamers import SerialStreamer
    from emager_py.utils.find_usb import find_psoc

    # choose_pictures()

    imgbox = ImageListbox()
    imgbox.start()

    port = find_psoc()
    streamer = SerialStreamer(port)


    def my_cb(gesture):
        print("Simulating long running process...")
        time.sleep(5)
        print(f"Gesture {gesture+1} done!")

    egt = EmagerGuidedTraining(
        streamer, resume_training_callback=my_cb, reps=3, training_time=4, callback_arg="gesture"
    )
    egt.start()
    print("Exiting...")
    
