import copy
import math
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import cv2
import pandas
import time
import requests

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch

import stereo_image_utils
from stereo_image_utils import get_detections, get_cost, draw_detections, annotate_class2 
from stereo_image_utils import get_horiz_dist_corner_tl, get_horiz_dist_corner_br, get_dist_to_centre_tl, get_dist_to_centre_br, get_dist_to_centre_cntr

#functions for the command handler

# def set_resolution(url: str, index: int=1, verbose: bool=False):
#     try:
#         if verbose:
#             resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
#             print("available resolutions\n{}".format(resolutions))

#         if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
#             requests.get(url + "/control?var=framesize&val={}".format(index))
#         else:
#             print("Wrong index")
#     except:
#         print("SET_RESOLUTION: something went wrong")

# def set_quality(url: str, value: int=1, verbose: bool=False):
#     try:
#         if value >= 10 and value <=63:
#             requests.get(url + "/control?var=quality&val={}".format(value))
#     except:
#         print("SET_QUALITY: something went wrong")

# def set_awb(url: str, awb: int=1):
#     try:
#         awb = not awb
#         requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
#     except:
#         print("SET_QUALITY: something went wrong")
#     return awb

# def set_angle(url: str, angle: int):
#     try:
#         requests.get(url + "/action?angle={}".format(angle))
#     except:
#         print("SET_ANGLE: something went wrong")


#This is the equivalent funciton to set_angle in javascript        
# function toggleCheckbox(angle) {
#     var xhr = new XMLHttpRequest();
#     xhr.open("GET", "/action?angle=" + angle, true);
#     xhr.send();
# }

def set_distance(url: str, dist: int):
    try:
        requests.get(url + "/action?distance={}".format(dist))
    except:
        print("SET_DISTANCE: something went wrong")


def set_speed(url: str, speed: int):
    try:
        requests.get(url + "/slider?value={}".format(speed))
    except:
        print("SET_SPEED: something went wrong")

#26 37 38

def object_upright(coords):
    return (abs(coords[0] - coords[2]) < abs(coords[1] - coords[3]))



#n, s, m, l, x
# see https://github.com/ultralytics/ultralytics for more information
model = YOLO("yolov8m.pt")
#class names
names =  model.model.names


#camera url. I've used a static url in the esp32 cam sketc.
# connecting through local network with url = 192.168.1.xxx
# might need to change this if you are connecting through an iphone hotspot or some other network
URL_left = "http://192.168.1.181"
URL_right = "http://192.168.1.129"
URL_car = "http://192.168.1.182"
AWB = True
cnt = 0
moved = False
total_angle = 0
brk = False
#focal length. Pre-calibrated in stereo_image_v6 notebook
fl = 2.043636363636363
tantheta = 0.7648732789907391-0.1


if __name__ == '__main__':
    # set_resolution(URL_left, index=10)
    # set_resolution(URL_right, index=10)
    cap_left = cv2.VideoCapture(1)

    cap_right = cv2.VideoCapture(2)
    # set_speed(URL_car, 230)
    # time.sleep(5)

    
    while True:
        mov_angle = []
        mov_dists = []
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        ### capture the images


        
#         if cap_left.isOpened():
#             ret_l, frame_l = cap_left.read()
#             #release the capture to stop a queu building up. I'm sure there are more efficient ways to do this.
#             cap_left.release()
            
#             if ret_l:
#                 cv2.imshow("left_eye", frame_l) 
# #             else:
# #                 cap_left.release()
# #                 cap_left = cv2.VideoCapture(URL_left + ":81/stream")

#         if cap_right.isOpened():
#             ret_r, frame_r = cap_right.read()
#             #release the capture to stop a queu building up. I'm sure there are more efficient ways to do this.
#             cap_right.release()

#             if ret_r:
#                 cv2.imshow("right_eye", frame_r) 
# #             else:
# #                 cap_right.release()
# #                 cap_right = cv2.VideoCapture(URL_right + ":81/stream")
        
        if ret_r and ret_l :
            imgs = [cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB),cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)]
            out_l = []
            out_r =[]
            #do stereo matching
            if cnt == 0:  #this condition added mostly for debugging
                out_l = (model.predict(source =cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB), save=False, conf = 0.4, save_txt=False, show = False ))[0]
                out_r = (model.predict(source =cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB), save=False, conf = 0.4, save_txt=False, show = False ))[0]
                
            #do stereo pair matching. See file below for details.
            # https://github.com/jonathanrandall/esp32_stereo_camera/blob/main/python_notebooks/stereo_image_v6.ipynb
            
            if (out_l.boxes.shape[0]<1 or out_r.boxes.shape[0]<1): #if I haven't detected anything move car and do another check
                # set_angle(URL_car, 10) ##move five degrees and check environment again.
                total_angle += 10
                continue
            
            if cnt == 0 and (out_l.boxes.shape[0]>0 and out_r.boxes.shape[0]>0): #cnt is just a control for debugging
                #boxes are the coordinates of the boudning boxes.
                cnt = 0 #1
                
                #find the image centre
                sz1 = frame_r.shape[1]
                centre = sz1/2

                #dets are bounding boxes and lbls are labels.
                det = []
                lbls = []
                
                #det[0] are the bounding boxes for the left image
                #det[1] are the bounding boxes for the right image

                if(out_l.boxes.shape[0]>0 and out_r.boxes.shape[0]>0):
                    det.append(np.array(out_l.boxes.xyxy))
                    det.append(np.array(out_r.boxes.xyxy))
                    lbls.append(out_l.boxes.cls)
                    lbls.append(out_r.boxes.cls)
                
                print(det)
                
                #get the cost of matching each object in the left image
                #to each object in the right image
                cost = get_cost(det, lbls = lbls,sz1 = centre)
                
                #choose optimal matches based on the cost.
                tracks = scipy.optimize.linear_sum_assignment(cost)                
                
                #find top left and bottom right corner distance to centre (horizonatlly)
                dists_tl =  get_horiz_dist_corner_tl(det)
                dists_br =  get_horiz_dist_corner_br(det)

                final_dists = []
                dctl = get_dist_to_centre_tl(det[0],cntr = centre)
                dcbr = get_dist_to_centre_br(det[0], cntr = centre)
                
                #measure distance of object from the centre so I can see how far I need to turn.
                d0centre = get_dist_to_centre_cntr(det[0], cntr = centre)
                d1centre = get_dist_to_centre_cntr(det[1], cntr = centre)
                
                #classes for left and right images. nm0 is left, nm1 is right
                q = [i.item() for i in lbls[0]]
                nm0 = [names[i] for i in q]
                q = [i.item() for i in lbls[1]]
                nm1 = [names[i] for i in q]
                
                #check if bottle is upright. height greater than width and move car certain angle.

                for i, j in zip(*tracks):
                    if (nm0[i])=='bottle':
                        print('is bottle')
                        #check if bottle is till upright
                        if object_upright(det[0][i]):
                            print('object upright')
#                             break
                            angle = (d0centre[i]+d1centre[j])/sz1*15 #15 worked well in experiments can play around with this.
                            # if objects are all the way to the right, then turn 15*2 30 degrees right
                            mov_angle.append(int(angle))
                            print(angle)
                        else:
                            print('object flat')
#                             break
                    if dctl[i] < dcbr[i]:
                        final_dists.append((dists_tl[i][j],nm0[i]))

                    else:
                        final_dists.append((dists_br[i][j],nm0[i]))
                
                #final distances as list
                fd = [i for (i,j) in final_dists]
                #find distance away
                dists_away = (7.05/2)*sz1*(1/tantheta)/np.array((fd))+fl
                cat_dist = []
                for i in range(len(dists_away)):
                    if (nm0[i])=='bottle':
                        mov_dists.append(dists_away[i])
                    cat_dist.append(f'{nm0[(tracks[0][i])]} {dists_away[i]:.1f}cm')
                    print(f'{nm0[(tracks[0][i])]} is {dists_away[i]:.1f}cm away')
                t1 = [list(tracks[1]), list(tracks[0])]
                frames_ret = []
                for i, imgi in enumerate(imgs):
                    img = imgi.copy()
                    deti = det[i].astype(np.int32)
                    draw_detections(img,deti[list(tracks[i])], obj_order=list(t1[1]))
                    annotate_class2(img,deti[list(tracks[i])],lbls[i][list(tracks[i])],cat_dist)
                    frames_ret.append(img)
                cv2.imshow("left_eye", cv2.cvtColor(frames_ret[0],cv2.COLOR_RGB2BGR))
                # cv2.imshow("right_eye", cv2.cvtColor(frames_ret[1],cv2.COLOR_RGB2BGR))
                # cv2.imshow("left_eye", frame_l)
                cv2.imshow("right_eye", frame_r)
                
                if (mov_dists and mov_dists[0] > 100): #don't move more than 100cm at this stage of testing.
                    continue
                
#                 if(not moved and mov_angle):
#                     set_angle(URL_car, mov_angle[0])
#                     time.sleep(2)
#                     if mov_angle[0] > 0:
#                         total_angle += mov_angle[0]
# #                     moved = True
#                     if(mov_dists):                       
#                         set_distance(URL_car, mov_dists[0]+3)
#                         time.sleep(2) ##wait two seconds, then reverse.
#                         set_distance(URL_car, -mov_dists[0]-3)
#                         time.sleep(2)
                
                # if (total_angle < 720): #two rounds
                #     set_angle(URL_car, 10) ##move five degrees and check environment again.
                #     total_angle += 10
                #     time.sleep(2)
                # else: 
                #     brk = True
                #     break
#                 while True:
#                     key1 = cv2.waitKey(1)
#                     if key1 == ord('p'):
#                         break
#                 key1 = cv2.waitKey(1)

            key = cv2.waitKey(1)

            if key == ord('r'):
                idx = int(input("Select resolution index: "))
                # set_resolution(URL, index=idx, verbose=True)

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                # set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)
                
            elif key == ord('p'):
                cnt = 0

            elif key == 27: #esc key
                print(out_l)
                break
    cv2.destroyAllWindows()
    cap_left.release()
    cap_right.release()

