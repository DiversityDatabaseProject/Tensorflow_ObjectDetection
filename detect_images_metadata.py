'''
This script opens images in a folder and writes metadata in csv file.
Author: Maria Rosario SEBASTIAN
Date:20/06/2022
'''
from PIL import Image
import csv, glob
import pandas as pd

def get_metadata(fn): 
    '''Opens an image, retrieves the image data, and puts the data into a dictionary'''
    i = Image.open(fn)
    fname=i.filename.replace("\\","/")
    ret = {
    "Filename": fname.split("/")[-1],
    "Size": i.size,
    "Height": i.height,
    "Width": i.width,
    "Format": i.format,
    "Mode": i.mode
    }
    return ret

def create_image_metadata(resultspath, csvpath):
    '''
    Writes metadata to csvpath on images from resultspath
    '''
    # Define path to images and grab all image filenames
    images = glob.glob(resultspath + '/*')

    #image data in each row
    rows=[]

    #header
    fieldnames=''
    for img in images:
        info_dict = get_metadata(img)
        fieldnames = info_dict.keys()
        rows.append(info_dict)

    with open(csvpath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_image_metadata(metadatapath, resultspath, csvpath):
    '''
    Reads image scraped metadata, transfers uploaded image metadata into different file.
    '''
    df = pd.read_csv(metadatapath)
    print('****************',df)

    #add header
    new_meta= pd.DataFrame(list(df.columns.values)).transpose()
    # Define path to images and grab all image filenames
    images = glob.glob(resultspath + '/*')

    #image data in each row
    rows=[]

    for img in images:
        filename=img.replace("\\","/")
        entry = df.loc[df['filename'] == filename.split("/")[-1]]
        rows.append(entry)


    new_meta = pd.concat(rows)
    print(new_meta)
    new_meta.to_csv(csvpath)



