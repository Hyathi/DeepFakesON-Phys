
import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser("Predict test videos")
parser.add_argument('input_dir', type=str, help="path to directory with videos")
args = parser.parse_args()
input_dir = args.input_dir

image_path = os.path.join(input_dir)
print(f"Processing videos in {image_path}")
image_name_video = []
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f in [f for f in os.listdir(image_path)]:
    print(f"Processing video: {f}")
    if not("_C.avi" in f): #OULU
        print(f"Skipping video {f}")
        continue
    print(f"Continued Processing video {f}")
    carpeta= os.path.join(image_path, f)
    cap = cv2.VideoCapture(carpeta)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(7)
    max_frames = int(nFrames)
    print(f"Processing video {f} with {max_frames} frames")
    ruta_parcial = os.path.join(input_dir, "DeepFrames")
    if not(os.path.exists(ruta_parcial)) :
        os.mkdir(ruta_parcial);
    ruta_parcial2 = os.path.join(input_dir, "RawFrames") 
    print(f"Saving raw frames to {ruta_parcial2}")
    if not(os.path.exists(ruta_parcial2)) :
        os.mkdir(ruta_parcial2);
    
    L = 36
    C_R=np.empty((L,L,max_frames))
    C_G=np.empty((L,L,max_frames))
    C_B=np.empty((L,L,max_frames))
    
    D_R=np.empty((L,L,max_frames))
    D_G=np.empty((L,L,max_frames))
    D_B=np.empty((L,L,max_frames))
    
    D_R2=np.empty((L,L,max_frames))
    D_G2=np.empty((L,L,max_frames))
    D_B2=np.empty((L,L,max_frames))
    
    medias_R = np.empty((L,L))
    medias_G = np.empty((L,L))
    medias_B = np.empty((L,L))
    
    desviaciones_R = np.empty((L,L))
    desviaciones_G = np.empty((L,L))
    desviaciones_B = np.empty((L,L))
    
    imagen = np.empty((L,L,3))
    
    medias_CR = np.empty((L,L))
    medias_CG = np.empty((L,L))
    medias_CB = np.empty((L,L))
    
    desviaciones_CR = np.empty((L,L))
    desviaciones_CG = np.empty((L,L))
    desviaciones_CB = np.empty((L,L))
    ka            = 1
    
    print(f"Before Cap is opened")
    while(cap.isOpened() and ka< max_frames):
        print(f"Processing frame {ka}")
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #rectangle around the faces
        for (x, y, w, h) in faces:
            # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
        print(f"Face shape {face.shape}")
       
        face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)
        # cv2.imshow('img', face)
        # cv2.waitKey()
        C_R[:,:,ka] = face[:,:,0]
        C_G[:,:,ka] = face[:,:,1]
        C_B[:,:,ka] = face[:,:,2]
        
        print(f"Frame {ka} processed")
        if ka > 1:
            D_R[:,:,ka-1] = ( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1] );
            D_G[:,:,ka-1] = ( C_G[:,:,ka] - C_G[:,:,ka-1] ) / ( C_G[:,:,ka] + C_G[:,:,ka-1] );
            D_B[:,:,ka-1] = ( C_B[:,:,ka] - C_B[:,:,ka-1] ) / ( C_B[:,:,ka] + C_B[:,:,ka-1] );
        ka = ka+1

    print(f"First set of loops")
    for i in range(0,L):
        for j in range(0,L):
            medias_R[i,j]=np.mean(D_R[i,j,:]) 
            medias_G[i,j]=np.mean(D_G[i,j,:]) 
            medias_B[i,j]=np.mean(D_B[i,j,:]) 
            desviaciones_R[i,j]=np.std(D_R[i,j,:]) 
            desviaciones_G[i,j]=np.std(D_G[i,j,:]) 
            desviaciones_B[i,j]=np.std(D_B[i,j,:]) 

    print(f"Second set of loops")
    for i in range(0,L):
        for j in range(0,L):
            medias_CR[i,j]=np.mean(C_R[i,j,:]) 
            medias_CG[i,j]=np.mean(C_G[i,j,:]) 
            medias_CB[i,j]=np.mean(C_B[i,j,:]) 
            desviaciones_CR[i,j]=np.std(C_R[i,j,:]) 
            desviaciones_CG[i,j]=np.std(C_G[i,j,:]) 
            desviaciones_CB[i,j]=np.std(C_B[i,j,:])         
            
    print(f"Third set of loops")
    for k in range(0,max_frames):
        D_R2[:,:,k] = (C_R[:,:,k] - medias_CR)/(desviaciones_CR+000.1)
        D_G2[:,:,k] = (C_G[:,:,k] - medias_CG)/(desviaciones_CG+000.1)
        D_B2[:,:,k] = (C_B[:,:,k] - medias_CB)/(desviaciones_CB+000.1)
     
    print(f"Fourth set of loops")
    for k in range(0,max_frames):
        
        imagen[:,:,0] = D_R2[:,:,k]
        imagen[:,:,1] = D_G2[:,:,k]
        imagen[:,:,2] = D_B2[:,:,k]

        imagen= np.uint8(imagen)
        
        nombre_salvar= os.path.join(ruta_parcial2,str(k)+'.png')
        cv2.imwrite(nombre_salvar, imagen)
        
    print(f"Fifth set of loops")
    for k in range(0,max_frames):
        
        D_R[:,:,k] = (D_R[:,:,k] - medias_R)/(desviaciones_R+000.1)
        D_G[:,:,k] = (D_G[:,:,k] - medias_G)/(desviaciones_G+000.1)
        D_B[:,:,k] = (D_B[:,:,k] - medias_B)/(desviaciones_B+000.1)
    
    print(f"Sixth set of loops")  
    for k in range(0, max_frames):
        # print(f"medias_R: {medias_R}")
        # print(f"desviaciones_R: {desviaciones_R}")

        # Avoid division by zero
        divisor_R = desviaciones_R + 0.1
        divisor_G = desviaciones_G + 0.1
        divisor_B = desviaciones_B + 0.1

        D_R[:,:,k] = (D_R[:,:,k] - medias_R) / divisor_R
        D_G[:,:,k] = (D_G[:,:,k] - medias_G) / divisor_G
        D_B[:,:,k] = (D_B[:,:,k] - medias_B) / divisor_B

        # Handle NaN or Inf values
        D_R[:,:,k] = np.nan_to_num(D_R[:,:,k], nan=0, posinf=0, neginf=0)
        D_G[:,:,k] = np.nan_to_num(D_G[:,:,k], nan=0, posinf=0, neginf=0)
        D_B[:,:,k] = np.nan_to_num(D_B[:,:,k], nan=0, posinf=0, neginf=0)

        imagen[:,:,0] = D_R[:,:,k]
        imagen[:,:,1] = D_G[:,:,k]
        imagen[:,:,2] = D_B[:,:,k]

        imagen = np.uint8(imagen)

        nombre_salvar = os.path.join(ruta_parcial, str(k) + '.png')
        cv2.imwrite(nombre_salvar, imagen)
        
    cap.release()
    cv2.destroyAllWindows()
print("Exiting...")
