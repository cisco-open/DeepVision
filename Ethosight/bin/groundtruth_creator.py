import os
import pandas as pd
import webbrowser
import requests
import cv2
import io
import base64
import warnings
import re

url_label_dict=dict()
warnings.filterwarnings("ignore")
path=r"/root/Ethosight/imagelinks_copy_t/" # Add your path to folder
os.mkdir("imageslinks_images")
path2=r"/root/Ethosight/imageslinks_images"
file_list=os.listdir(path)
i=0
os.chdir(path)
for file in file_list:
    with open(file,"r") as f:
        urls_in_file=re.split("\n", f.read())
        for line in urls_in_file:
            i+=1
            print("############################# Current Ground Truth : ",file,"  url count:",i,"####################################")
            filename = str(i)+'.jpg'
            try:
                if line.startswith("data:image"):
                    url_data = line.split(',')[1]
                    os.chdir(path2)
                    with open(filename, "wb") as f2:
                        f2.write( base64.b64decode(url_data))
                    os.chdir(path)
                else:
                    response = requests.get(line,stream=True)
                    os.chdir(path2)
                    with open(filename, 'wb') as out_file:
                        out_file.write(response.content)
                    os.chdir(path)
                url_label_dict.update({line:file[:-4]})
                    
                os.chdir(path2)
                cv2.imshow(file[:-4],cv2.imread(filename))
                cv2.waitKey(0)
                cv2.destroyAllWindows()    
                label_change=str(input("\n\n ########## Enter new label for the image ######\n"))
                print("\n")
                
                if label_change=="":
                    continue
                else:        
                    os.chdir(path)
                    url_label_dict[line]=label_change.strip()
            except:
                continue
            
            
print(url_label_dict)
os.chdir("/root/Ethosight/") 
os.mkdir("modified_image_links")
os.chdir("/root/Ethosight/modified_image_links")
for url,new_labels in url_label_dict.items():
    print(new_labels)
    if new_labels+".txt" not in os.listdir("/root/Ethosight/modified_image_links"):
        print("yes")
        with open(new_labels,"w+")as f_:
            f_.write(url)
        
    else:
        print("no")
        with open(new_labels+".txt","a")as writ:
            writ.write(url)
    
        
                
            
            