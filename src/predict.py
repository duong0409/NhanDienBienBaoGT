import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import os

# Load the trained model to classify signs
model = load_model('my_model.h5')

# Dictionary to label all traffic signs class.
classes = {1: 'Giới hạn tốc độ (20km/h)',
           2: 'Giới hạn tốc độ (30km/h)',
           3: 'Giới hạn tốc độ (50km/h)',
           4: 'Giới hạn tốc độ (60km/h)',
           5: 'Giới hạn tốc độ (70km/h)',
           6: 'Giới hạn tốc độ (80km/h)',
           7: 'Hết giới hạn tốc độ (80km/h)',
           8: 'Giới hạn tốc độ (100km/h)',
           9: 'Giới hạn tốc độ (120km/h)',
           10: 'Cấm vượt',
           11: 'Cấm vượt xe có trọng tải trên 3,5 tấn',
           12: 'Quyền ưu tiên tại ngã tư',
           13: 'Đường ưu tiên',
           14: 'Nhường đường',
           15: 'Dừng lại',
           16: 'Cấm xe vào',
           17: 'Cấm xe > 3,5 tấn',
           18: 'Đường cấm',
           19: 'Cảnh báo chung',
           20: 'Rẽ trái nguy hiểm',
           21: 'Đường cong nguy hiểm bên phải',
           22: 'Đường cong đôi',
           23: 'Đường gập ghềnh',
           24: 'Đường trơn trượt',
           25: 'Đường hẹp bên phải',
           26: 'Công trình đường bộ',
           27: 'Tín hiệu giao thông',
           28: 'Người đi bộ',
           29: 'Trẻ em băng qua đường',
           30: 'Xe đạp băng qua đường',
           31: 'Cẩn thận băng/tuyết',
           32: 'Động vật hoang dã băng qua đường',
           33: 'Tốc độ kết thúc + giới hạn vượt',
           34: 'Rẽ phải phía trước',
           35: 'Rẽ trái phía trước',
           36: 'Chỉ đi phía trước',
           37: 'Đi thẳng hoặc rẽ phải',
           38: 'Đi thẳng hoặc rẽ trái',
           39: 'Giữ bên phải',
           40: 'Giữ bên trái',
           41: 'Vòng xuyến phía trước',
           42: 'Kết thúc lệnh cấm vượt',
           43: 'Kết thúc lệnh cấm vượt xe > 3,5 tấn'}

# Initialize GUI
top = tk.Tk()
top.geometry('900x700')
top.title('Nhận diện biển báo giao thông')
top.configure(background='#f4f4f9')

# Header
header = Label(top, text="Nhận diện biển báo giao thông", font=('Arial', 24, 'bold'), bg='#4c77a6', fg='white', pady=10)
header.pack(fill=X)

# Create and place the buttons frame
button_frame = Frame(top, bg='#f4f4f9')
button_frame.pack(pady=20)

upload_button = Button(button_frame, text="Chọn ảnh", command=lambda: upload_image(), bg='#1e81b0', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=10)
upload_button.grid(row=0, column=0, padx=20)

webcam_button = Button(button_frame, text="Nhận diện bằng webcam", command=lambda: run_webcam(), bg='#1e81b0', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=10)
webcam_button.grid(row=0, column=1, padx=20)

# Image display and result label
image_frame = Frame(top, bg='#f4f4f9')
image_frame.pack(pady=20)

sign_image = Label(image_frame, bg='#d9e4f5', width=400, height=300, relief='groove', bd=2)
sign_image.pack()

label = Label(top, text="", font=('Arial', 14, 'italic'), bg='#f4f4f9', fg='#333')
label.pack(pady=10)

# Functions
def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    sign = classes[predicted_class + 1]

    label.configure(text=f"Kết quả: {sign}", fg='#1e81b0')


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 300))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im

        classify(file_path)


def run_webcam():
    os.system("python webcam.py")

# Main loop
top.mainloop()
