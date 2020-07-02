"""
	dibuat dan dikembangkan oleh sscompany
	dapat disebar luaskan tanpa di pungut biaya
	dapat dikembangan oleh siapapun

	create by : muhamad safrudin abdul azis
	email : karyaanakdesa632@gmail.com
"""
import cv2
import tkinter.messagebox
import numpy as np
import os
from tkinter import Tk,PhotoImage
from tkinter.ttk import Label
import threading
from PIL import Image

#membuat fungsi untuk mengecek apakah folder telah ada
def cek_folder(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#membuat fungsi untuk mendapatkan daftar user dan id user
def get_user_id():
    #cek apakah file data_user.txt ada
    if os.path.isfile("training/data_user.txt"):
        #membuka file untuk dibaca
        file = open("training/data_user.txt","r")
        isi = file.readlines()
        data = []

        #decode file
        for baris in isi:
            baris = baris.split("\n")
            baris = baris[0]
            baris = baris.split(":")
            data.append(baris)

        return data
    else:
        isi = [0,"none"]
        data = []
        data.append(isi)
        return data
        print("Belum ada user yang dimasukkan")

#membuat class untuk setting gui
class setting_gui:
    def __init__(self,root):
        self.root = root
        self.root.title("Deteksi Wajah | SS Company 2019")#setting judul di window
        self.root.geometry("500x500")#mengatur labay layar atau window
        self.root.resizable(False,False)#mengatur agar window tidak bisa diperlebar
        self.root.protocol('WM_DELETE_WINDOW',self.quit)

        #memanggil fungsi untuk atur komponen di window
        self.atur_komponen()
        self.deklarasi_variable()

        #memanggil fungsi untuk merekam gambar via webcam
        t = threading.Thread(target=self.rekam, args=())
        t.start()

    #fungsi untuk deklarasi semua variable global yang akan digunakan
    def deklarasi_variable(self):
        self.proses = True

    #fungsi untuk setting semua komponen yang akan ditampikan
    def atur_komponen(self):
        #membuat tempat untuk menampilkan gambar
        self.image = Label(self.root)
        self.image.place(x=0,y=0)
        self.image['text'] = "Loading"

    def rekam(self):
        #membuat fungsi untuk mencari wajah yang sama
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        #cek apakah folder training sudah ada
        cek_folder("training/")

        #membaca database wajah
        if os.path.isfile("training/hasil.yml"):
            recognizer.read("training/hasil.yml")
        else:
            tkinter.messagebox.showinfo("Pemberitahuan","Belum ada wajah yang diinput!")

        #memanggil fungsi untuk deteksi wajah
        face = cv2.CascadeClassifier("xml/deteksi_wajah.xml")

        #setting font untuk ditampilkan pada window
        font = cv2.FONT_HERSHEY_SIMPLEX

        #mulai mengambil gambar di kamera
        cp = cv2.VideoCapture(0)

        #mulai mendeteksi wajah
        while self.proses:
            #membaca hasil rekaman kamera
            _, img = cp.read()

            #merubah gambar menjadi gray scale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #mencari semua wajah di gambar
            muka = face.detectMultiScale(gray, 1.2, 5)

            #membaca setiap wajah yang ditemukan
            for (x,y,w,h) in muka:
                #membuat kotak untuk wajah
                cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

                #mengurai wajah untuk mendapatkan id dan nama user
                Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                data = get_user_id()
                for isi in data:
                    if int(isi[0]) == Id:
                        Id = isi[1] + " {0:.2f}%".format(round(100 - confidence, 2))

                cv2.rectangle(img, (x - 22, y - 90), (x + w + 22, y - 22), (255, 255, 255), -1)
                cv2.putText(img, str(Id), (x, y - 40), font, 1, (0, 0, 0), 3)

            #menampilkan gambar di window
            cek_folder("log/")
            cv2.imwrite("log/deteksi.png",img)
            pic = Image.open("log/deteksi.png")
            pic = pic.resize((500, 500), Image.ANTIALIAS)
            pic.save("log/input.png")
            gambar = PhotoImage(file="log/input.png")
            self.image.config(image=gambar)
            self.image.image = gambar

    #membaut fungsi untuk meminta konfirmasi sebelum exit
    def quit(self):
        self.proses = False
        cek = tkinter.messagebox.askquestion("Peringatan","Apakah kamu akan keluar?")
        if cek == "yes":
            self.root.destroy()
        else:
            self.proses = True
            t = threading.Thread(target=self.rekam,args=())
            t.start()


#menampilkan form
root = Tk()
setting_gui(root)
root.mainloop()
