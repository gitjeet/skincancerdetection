from selenium import webdriver
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy
#load the trained model to classify sign
from keras.models import load_model
from keras.preprocessing import image as image1
import requests
from bs4 import  BeautifulSoup
import pandas
import csv
from tkinter import ttk
model = load_model(r'C:\Users\Abhijeet\Desktop\Skin dataset\my_modelCNN.h5')
with open(r'C:\Users\Abhijeet\Desktop\Skin dataset\train\benign\model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

with open(r'C:\Users\Abhijeet\Desktop\Skin dataset\train\benign\modelnaive.pkl', 'rb') as f1:
    clf3 = pickle.load(f1)
with open(r'C:\Users\Abhijeet\Desktop\Skin dataset\train\benign\modelrandomforest.pkl', 'rb') as f4:
    clf4 = pickle.load(f4)
classes = { 1:'Malignants',
            2:'Benigns' }
i=0

top=tk.Tk()
top.geometry('800x600')
top.title('Skin Cancer Hackathon')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
label1=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
     global label_packed
     image = Image.open(file_path)
     image = cv2.imread(file_path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     

     image = cv2.resize(image, (96, 96))
     
     image = image1.img_to_array(image)
     image = image.reshape(1,96,96,1)
     image_knn =image
     image_knn=image.reshape(1,96*96)
     pred = model.predict_classes([image])[0]
     sign = classes[pred+1]
     print(sign)
     pred_knn=clf2.predict(image_knn)
     sign_knn=classes[pred+1]
     print(sign_knn)
     pred_naive=clf3.predict(image_knn)
     sign_naive=classes[pred+1]
     print(sign_naive)
     pred_randomforest=clf4.predict(image_knn)
     sign_random =classes[pred+1]
     print(sign_random)
     label.configure(foreground='#011638', text="According to 1.CNN  "+ sign +" 2. RandomForest " + sign_random  +" 3. KNN "+ sign_knn +" 4. NaiveBayes " +sign_naive )
     if (sign == sign_knn==sign_naive==sign_random=='Malignants'):
         
         make_app()
         
     else:
         label1.configure(foreground="#011638",text="Skin Type Bengins You are safe")
         label1.pack(side = TOP,expand=True)
         

         
    
def show_classify_button(file_path):
     classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
     classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
     classify_b.place(relx=0.79,rely=0.46)

def upload_image():
     try:
         file_path=filedialog.askopenfilename()
         uploaded=Image.open(file_path)
         uploaded.thumbnail(((top.winfo_width()/2.25),        (top.winfo_height()/2.25)))
         im=ImageTk.PhotoImage(uploaded)

         sign_image.configure(image=im)
         sign_image.image=im
         label.configure(text='')
         show_classify_button(file_path)
         label1.configure(text='')
     except:
         pass

flag=0
def new_window(): 
    # window =tk.Tk()
    # window.title("Making Appointment")
    
    def webscrapp(USER_INP):
        print(USER_INP)   
        k=USER_INP
        print(k)
        r=requests.get("https://www.practo.com/"+k+"/dermatologist")
        c=r.content
        soup = BeautifulSoup(c,"html.parser")
        all=soup.find_all("div",{"class":"listing-doctor-card"})
        all[0].find("h2",{"class":"doctor-name"}).text
        l=[]
        for item in all:
            d={}
            d["Doctor Name"] =       (item.find("h2",{"class":"doctor-name"}).text)
            d["No.of Experience in Years "]=(item.find("div",{"class":"uv2-spacer--xs-top"}).find("div").text.replace('\xa0years experience overall',""))
            d["Consultation Fees"] =(item.find("span",{"class":"","data-qa-id":"consultation_fee"}).text.replace("₹","Rs"))
            try:
                d["Rating"] = (item.find("span",{"data-qa-id":"doctor_recommendation"}).text)
            except:
                d["Rating"]=("None")
            l.append(d)
         
        df =pandas.DataFrame(l)
        print(df)
        df.to_csv("Output[i].csv")
        root = Tk()
        root.title("Doctors available")
        width = 500
        height = 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        root.geometry("%dx%d+%d+%d" % (width, height, x, y))
        root.resizable(0, 0)
        
        TableMargin = Frame(root, width=500)
        TableMargin.pack(side=TOP)
        
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=("Doctor Name", "No.of Experience in Years ", "Consultation Fees",'Rating'), height=400, selectmode="extended",
                            yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        tree.heading('Doctor Name', text="Doctor Name", anchor=W)
        tree.heading('No.of Experience in Years ', text="No.of Experience in Years ", anchor=W)
        tree.heading('Consultation Fees', text="Consultation Fees", anchor=W)
        tree.heading('Rating', text="Rating", anchor=W)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.column('#1', stretch=NO, minwidth=0, width=120)
        tree.column('#2', stretch=NO, minwidth=0, width=120)
        tree.column('#3', stretch=NO, minwidth=0, width=120)
        tree.column('#3', stretch=NO, minwidth=0, width=120)
        tree.pack()
        with open(r'C:\Users\Abhijeet\Desktop\Skin dataset\train\benign\Output[i].csv') as f:
          reader = csv.DictReader(f, delimiter=',')
          for row in reader:
            emp_id = row['Doctor Name']
            exp = row['No.of Experience in Years ']
            cf = row['Consultation Fees']
            ra = row['Rating']
            tree.insert("", 0, values=(emp_id, exp, cf,ra))
        new_window2()
        root.mainloop()
        
    USER_INP = simpledialog.askstring(title="Appointment",
                                  prompt="Enter Your Location:")
    webscrapp(USER_INP)
          
    # window.mainloop() 
       
   
def make_app():
    global flag
    if(flag==0):
             
        flag=1
        new1=Button(top,text="Make an Appoitment",command=new_window)
        new1.pack(side=BOTTOM,expand=True)
        new1.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    else:
        pass


def new_window2():
    window2=tk.Tk()
    window2.title("Making Appointment")
    def message():


        driver = webdriver.Chrome(r'C:\Users\Abhijeet\Desktop\chromedriver (1)')
        driver.get("https://web.whatsapp.com/")
        driver.maximize_window()
        
        name = e2.get()
        
        name1= e1.get()
        
        
        msg = ("Hello doctor "+name +" my name is " + name1+" i have been checked with malignant skin disease using AI and ML models and want to make an appointment with you ")
        print(msg)
        
        
        
        messagebox.showinfo("showinfo", "This Message will be sent \n" + msg ) 
        count = int(1)
        
        
        
        user = driver.find_element_by_xpath("//span[@title='{}']".format(name))
        user.click()
        
        msg_box = driver.find_element_by_xpath("//*[@id='main']/footer/div[1]/div[2]/div/div[2]")
        
        for index in range(count):
            msg_box.send_keys(msg)
            driver.find_element_by_xpath("//*[@id='main']/footer/div[1]/div[3]/button").click()
        
        messagebox.showinfo("Appointment done", "Success !! Message Sent") 
        
        
        
    
    b1= Button(window2,text="Enter your name",command=message)
    b1.grid(row=0,column=0)
    b2 =Button(window2,text="Enter Doctor Name")
    b2.grid(row=1 ,column=0)
    
    
    e1=Entry(window2)
    e1.grid(row=0,column=1)    
    e2=Entry(window2)
    e2.grid(row=1,column=1)
    window2.mainloop()
       
    
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)

sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Skin Cancer Detection",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
i+=1
top.mainloop()