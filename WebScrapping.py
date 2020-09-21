import requests
from bs4 import  BeautifulSoup
print("Enter the your Location")
k= input().lower()

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
import pandas 
df =pandas.DataFrame(l)
print(df)
df.to_csv("Output.csv")