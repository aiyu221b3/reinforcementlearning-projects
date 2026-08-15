import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

hs=mp.solutions.hands
hd=mp.solutions.drawing_utils
hands=hs.Hands(max_num_hands=1)
sw,sh=pyautogui.size()

sm=5
fr=100
bt=15
cc=(0,255,0)
px=py=0
mx=my=0
t1=t2=0
cd=.25
c=None

while True:
    ok,img=cap.read()
    if not ok:
        break
    img=cv2.flip(img,1)
    h,w=img.shape[:2]

    if c is None or c.shape[:2]!=(h,w):
        c=np.zeros((h,w,3),np.uint8)

    r=hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    if r.multi_hand_landmarks:
        lm=r.multi_hand_landmarks[0]
        hd.draw_landmarks(img,lm,hs.HAND_CONNECTIONS)

        p=np.array([[x.x*w,x.y*h] for x in lm.landmark],dtype=np.float32)
        p=np.rint(p).astype(np.int32)
        i,m,t=p[[8,12,4]]
        up=p[[8,12],1]<p[[6,10],1]
        x,y=i
        now=time.monotonic()

        cv2.rectangle(img,(fr,fr),(w-fr,h-fr),(255,0,255),2)

        d1=np.sum((i-t)**2)
        d2=np.sum((m-t)**2)

        if d1<900 and now-t1>cd:
            cv2.circle(img,tuple(i),10,(0,255,0),cv2.FILLED)
            pyautogui.click()
            t1=now

        if d2<900 and now-t2>cd:
            cv2.circle(img,tuple(m),15,(0,0,255),cv2.FILLED)
            pyautogui.doubleClick()
            t2=now

        if up[0] and up[1]:
            px=py=0
            cv2.rectangle(img,(x-15,y-25),(m[0]+15,m[1]+25),cc,cv2.FILLED)
            q=np.array([
                np.interp(x,(fr,w-fr),(0,sw)),
                np.interp(y,(fr,h-fr),(0,sh))
            ])
            mx+=(q[0]-mx)/sm
            my+=(q[1]-my)/sm
            pyautogui.moveTo(mx,my)
        elif up[0]:
            cv2.circle(img,(x,y),15,cc,cv2.FILLED)
            if px==0 and py==0:
                px,py=x,y
            cv2.line(c,(px,py),(x,y),cc,bt)
            px,py=x,y
        else:
            px=py=0

    mask=np.any(c,axis=2)
    img[mask]=c[mask]

    cv2.imshow("Neural Interface",img)
    if cv2.waitKey(1)&0xFF==ord("s"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
