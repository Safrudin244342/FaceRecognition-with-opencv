"""
	dibuat dan dikembangkan oleh sscompany
	dapat disebar luaskan tanpa di pungut biaya
	dapat dikembangan oleh siapapun

	create by : muhamad safrudin abdul azis
	email : karyaanakdesa632@gmail.com
"""
from tkinter import Tk,PhotoImage
from tkinter.ttk import Label,Entry,Button,Progressbar
import cv2
import threading
from PIL import Image
import os
import tkinter.messagebox
import numpy as np


#membuat fungsi untuk mengecek apakah folder telah ada
def cek_folder(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#membuat fungsi untuk mengambil gambar dari folder image
def get_image(path,deteksi):
    # mendapatkan semua gambar dari folder image
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        #conver image ke graysclae agar mudah di proses
        PIL_img = Image.open(imagePath).convert('L')

        #merubah image ke numpy array
        img_numpy = np.array(PIL_img, 'uint8')

        #mendapatkan id dari gambar
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        #mendeteksi wajah di gambar
        faces = deteksi.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

class setting_gui():
    def __init__(self,root):
        self.root = root
        self.root.title('Input wajah | SS Company 2019')
        self.root.resizable(False,False)#mengatur agar windows tidak bisa diperbesar atau perkecil
        self.root.geometry("550x350")#mengatur lebay layar
        self.root.protocol('WM_DELETE_WINDOW',self.quit)

        self.deklarasi_variable()
        self.atur_komponen()

        #menangkap gambar
        t = threading.Thread(target=self.tangkap_gambar, args=())
        t.start()

        self.get_id_user()

    def deklarasi_variable(self):
        self.berjalan = True#deklarasi variable untuk menunjukan program masih berjalan
        self.simpan = False
        self.count = 0
        self.akhir = 100#jumlah foto yang akan diambil
        self.proses = 0

    def atur_komponen(self):
        #tempat untuk menampilkan gambar
        self.image = Label(self.root)
        self.image.place(x=10,y=10)

        #textbox untuk gambar
        self.txt_gambar = Label(self.root)
        self.txt_gambar.place(x=80,y=315)

        #membuat form untuk memasukkan nama
        #entry nama
        Label(self.root,text="Masukkan Nama Anda : ").place(x=340,y=10)
        self.txt_nama = Entry(self.root)
        self.txt_nama.place(x=340,y=30)

        #button untuk daftar wajah
        self.btn_daftar = Button(self.root,text="daftar",command=self.daftar).place(x=340,y=60)

        #komponen progres bar untuk memberi tahu kapan proses selesai
        Label(self.root,text="Proses Sedang berjalan : ").place(x=340,y=100)
        self.proses_bar = Progressbar(self.root,orient='horizontal',length=180,mode='determinate')
        self.proses_bar.place(x=340,y=120)

    def tangkap_gambar(self):
        cp = cv2.VideoCapture(0)
        while self.berjalan:
            #membaca input dari webcam
            _,img = cp.read()

            #mengimport file pemindai wajah
            face = cv2.CascadeClassifier('xml/deteksi_wajah.xml')

            #merubah gambar menjadi gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            muka = face.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in muka:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if len(muka) == 1 and self.simpan and self.proses <= self.akhir:#jika perintah simpan telah diterima
                    self.simpan_wajah(gray,x,y,w,h)#memanggil fungsi simpan wajah
                    self.update_proses_bar(self.proses)#memanggil fungsi update status bar
                    self.proses += 1
                if self.proses == self.akhir and self.simpan:#jika proses penambilan gambar telah selesai
                    self.decode_gambar()
                    self.simpan = False

            #jika tidak ditemukan wajah pada gambar
            if len(muka) == 0:
                self.txt_gambar['text'] = "Wajah tidak ditemukan!"
            elif len(muka) == 1:
                self.txt_gambar['text'] = "Wajah ditemukan"
            elif len(muka) > 1:
                self.txt_gambar['text'] = "Ditemukan " + str(len(muka)) + " wajah"

            cek_folder("log/")
            cv2.imwrite("log/input.png",img)
            pic = Image.open("log/input.png")
            pic = pic.resize((300,300),Image.ANTIALIAS)
            pic.save("log/input.png")
            gambar = PhotoImage(file="log/input.png")
            self.image.config(image=gambar)
            self.image.image = gambar

    #fungsi untuk btn_daftar
    def daftar(self):
        #cek apakah folder image telah ada
        cek = tkinter.messagebox.askquestion("Pemberitahuan","Pengambilan gambar akan dimulai\nApakah kamu mau lanjut?")
        if cek == 'yes':
            cek_folder("image/")
            self.simpan = True
            self.count = 0
            self.proses = 0

    #fungsi untuk menyimpan gambar ke folder image
    def simpan_wajah(self,gray,x,y,w,h):
        id = self.get_id_user()#id untuk user yang akan disimpan wajah nya
        self.count += 1
        cv2.imwrite("image/User." + str(id) + '.' + str(self.count) + ".png", gray[y:y + h, x:x + w])
        print("gambar ke " + str(self.proses) + " disimpan")

    #fungsi untuk mengurai gambar
    def decode_gambar(self):
        cek_folder('training/')
        tkinter.messagebox.showinfo("Pemberitahuan","Proses Pengambilan Gambar telah Selesai\n"
                                                    "Dilanjutkan proses training\n"
                                                    "Jangan melakukan aksi apapun")
        recognizer = cv2.face.LBPHFaceRecognizer_create()#deklarasi untuk pengenalan wajah
        deteksi = cv2.CascadeClassifier('xml/deteksi_wajah.xml')#import file untuk deteksi wajah

        #mendapatkan gambar dan id user
        muka,id = get_image("image",deteksi)

        #decode semua gambar
        recognizer.train(muka,np.array(id))

        #menyimpan hasil proses
        recognizer.save("training/hasil.yml")

        # membuat database nama dan id gambar
        self.simpan_user_id()

        tkinter.messagebox.showinfo("Berhasil","Wajah telah ditambahkan")#menampilkan pesan pemberitahuan bahwa proses te;ah berhasil
        self.proses_bar['value'] = 0#update status proses bar

    #menampilkan presentasi proses yang sedang berjalan
    def update_proses_bar(self,proses):
        presentase = int((proses/100) * 100)
        self.proses_bar['value'] = presentase

    #fungsi untuk menyimpan nama user dan id gambar yang terhubung
    def simpan_user_id(self):
        #cek apakah file telah ada
        if os.path.isfile('training/data_user.txt'):
            #jika file sudah ada
            file = open('training/data_user.txt','a')
            text = "\n"
        else:
            #jika file belum ada
            file = open("training/data_user.txt",'w')
            text = ""

        #mendapatkan id terakhir yang telah tersimpan
        id = self.get_id_user()
        nama = self.txt_nama.get()

        #menuliskan nama ke data_user.txt
        text = text + str(id) + ":" + nama
        file.write(text)

    def get_id_user(self):
        # cek apakah file telah ada
        if os.path.isfile('training/data_user.txt'):
            #jika file sudah ada
            file = open('training/data_user.txt', 'r')
        else:
            #jika file belum ada
            return 0;

        #membaca file perbaris
        baris = file.readlines()
        if len(baris) > 0:
            baris = baris[len(baris) - 1]
            baris = baris.split("\n")
            baris = baris[0]
            id = baris.split(":")
            id = int(id[0]) + 1
        else:
            id = 0
        return id

    def hentikan_proses(self):
        self.berjalan = False
        cek = tkinter.messagebox.askquestion("pemberitahuan","Apakah Kamu akan mengakhiri proses ini?")
        if cek == "yes":
            self.root.quit()
        else:
            self.berjalan = True
            t = threading.Thread(target=self.tangkap_gambar,args=())
            t.start()

    def quit(self):
        self.berjalan = False
        cek = tkinter.messagebox.askquestion("pemberitahuan", "Apakah Kamu akan mengakhiri proses ini?")
        if cek == "yes":
            self.root.destroy()
        else:
            self.berjalan = True
            t = threading.Thread(target=self.tangkap_gambar, args=())
            t.start()

root = Tk()
setting_gui(root)
root.mainloop()
