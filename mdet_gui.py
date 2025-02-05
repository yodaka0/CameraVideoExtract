import os
import tkinter as tk
from tkinter import filedialog, messagebox
from natsort import natsorted
from exec_mdet import ExecMdet


def create_new_structure(src_dir):
    dst_dir = os.path.dirname(src_dir)
    for dir, _ ,_ in os.walk(src_dir):
        dirs_name = dir.replace(dst_dir, "")
        mid_dir = dirs_name.replace(os.path.sep, "_out" + os.path.sep)  + "_out"
        mid_dir = mid_dir.lstrip("/")
        mid_dir = mid_dir.lstrip("\\")
        new_dir = os.path.join(dst_dir, mid_dir)
        new_dir = os.path.normpath(new_dir)
        os.makedirs(new_dir, exist_ok=True)


def find_video_files(folder_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    video_files = natsorted(video_files)

    return video_files


def browse_session_root():
    directory = filedialog.askdirectory()
    if directory:
        session_root_entry.delete(0, tk.END)
        session_root_entry.insert(0, directory)

def start_process():
    session_root = session_root_entry.get()
    threshold = float(threshold_entry.get())
    checkpoint = checkpoint_entry.get()
    video_separaion = video_frams.get()
    skip = skip_swich.get()
    md_model = model_var.get()
    open_folder = True

    image_files = find_video_files(session_root)
     
    if checkpoint.startswith("r"):
        checkpoint = len(image_files) // int(checkpoint[1:])
    elif not checkpoint:
        checkpoint = None
    else:
        checkpoint = int(checkpoint)

    print(f"Session Root:{session_root}, Threshold:{threshold}, Checkpoint:{checkpoint}, Video separation interval::{video_separaion}, Exist skip:{skip}")

    
    #parent_dir = os.path.dirname(session_root)
    create_new_structure(session_root)
    
    output_dir = session_root + "_out"

    exec_mdet = ExecMdet(image_files, threshold, session_root, checkpoint, video_separaion, skip, md_model)
    exec_mdet.run_detector_with_image_queue()
    messagebox.showinfo("Info", f"Process completed successfully. Check the output folder {output_dir} for results.")
    root.destroy()
    if open_folder:
        os.startfile(output_dir)


root = tk.Tk()
root.title("ImageExtractWin GUI")

tk.Label(root, text="Session Root:").grid(row=0, column=0, padx=10, pady=5)
session_root_entry = tk.Entry(root, width=50)
session_root_entry.grid(row=0, column=1, padx=10, pady=5)
browse_button = tk.Button(root, text="Browse", command=browse_session_root)
browse_button.grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Threshold:").grid(row=1, column=0, padx=10, pady=5)
threshold_entry = tk.Entry(root, width=50)
threshold_entry.insert(0, "0.2") 
threshold_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Checkpoint:").grid(row=2, column=0, padx=10, pady=5)
checkpoint_entry = tk.Entry(root, width=50)
checkpoint_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Video separation interval:").grid(row=3, column=0, padx=10, pady=5)
video_frams = tk.Entry(root, width=50)
video_frams.insert(0, "1")
#set interval between frames
video_frams.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Skip existing files:").grid(row=4, column=0, padx=10, pady=5)
skip_swich = tk.BooleanVar(root)
skip_swich.set(False)
#add checkbutton for skip existing files
skip_checkbutton = tk.Checkbutton(root, variable=skip_swich)
skip_checkbutton.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Select MegaDetector's model:").grid(row=5, column=0, padx=10, pady=5)
model_var = tk.StringVar(root)
model_var.set("MegaDetector_v6c")
model_option = tk.OptionMenu(root, model_var, "MegaDetector_v5","HerdNet","MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c")
model_option.grid(row=5, column=1, padx=10, pady=5)


start_button = tk.Button(root, text="Start", command=start_process)
start_button.grid(row=6, column=0, columnspan=3, pady=20)

if __name__ == "__main__":
    root.mainloop()

