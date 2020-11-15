import streamlit as st
import numpy as np 
import cv2
import os 
from PIL import Image, ImageEnhance
import easyocr
import pytesseract

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

def main():


    st.title("Invoice Dashboard")
    activities = ["Invoice Processing"]
    choice = st.sidebar.selectbox("Select Option", activities)


    if choice == 'Invoice Processing':
        st.subheader("Invoice Image Preprocessing")

        #upload Image
        image_file = st.file_uploader("Upload Image", type = ["jpg", "png", 'jpeg'])

        ##Display Original Image 
        if image_file is not None:
            out_image = Image.open(image_file)
            st.text("Uploaded Image")
            # st.image(our_image, use_column_width=True)

        #Left side panel options 
        enhance_type = st.sidebar.radio("Enhance Type", ["Original","Processed","Get Co-ordinates"])

        if enhance_type == "Original":
            #display image as it is 
            st.image(out_image, use_column_width=True)

        #change Threshold
        elif enhance_type == 'Processed':
            new_img = np.array(out_image.convert("RGB"))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            x = st.slider("Change Threshold Value", min_value = 10, max_value=255)
            ret, thresh1 = cv2.threshold(gray, x, 255, cv2.THRESH_BINARY)
            st.image(thresh1, use_column_width=True

            #show only the boundary
            if st.button("Detect Outlay"):
                img_bin = 255-thresh1
                kernel_length = np.array(img_bin).shape[1]//80
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                vert_struct = cv2.erode(img_bin, vertical_kernel, iterations=3)
                vert_img = cv2.dilate(vert_struct, vertical_kernel, iterations=3)
                horiz_struct = cv2.erode(img_bin, horizontal_kernel, iterations=1)
                horiz_img = cv2.dilate(horiz_struct, horizontal_kernel, iterations=1)
                alpha = 0.5
                beta = 1.0 - alpha
                combined_image = cv2.addWeighted(vert_img, alpha, horiz_img, beta, 0.0)
                combined_image = cv2.erode(~combined_image, kernel, iterations=2)
                (thresh,combined_image) = cv2.threshold(combined_image, 128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                st.image(combined_image, use_column_width=True)
      
                
        #show the co-ordinates
                elif enhance_type == "Get Co-ordinates":
                    imgcpy = combined_image.copy()
                    output = pytesseract.image_to_data(imgcpy, output_type=Output.DICT)
                    n_boxes = len(output['level'])
                    for i in range(n_boxes):
                        (x,y,w,h)=  (output['left'][i], output['top'][i], output['width'][i], output['height'][i])
                        img2 = cv2.rectangle(thresh1, (x,y), (x+w, y+h), (255,0,0),1)
                        st.text_area([x,y, x+w, y+h])



                







        







if __name__ == '__main__':
    main()