import cv2 as cv
import mediapipe as mp
# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

capture= cv.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    isTrue, image = capture.read()
    
    # flip image
    image = cv.cvtColor(cv.flip(image,1),cv.COLOR_BGR2RGB)
    
    # store result
    result = hands.process(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw dots and lines on hand
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        # For detecting Left Right Or Both Hands
        if len(result.multi_handedness) == 2: 
			# Display 'Both Hands' on the image
            cv.putText(image, 'Both Hands', (250, 50),cv.FONT_HERSHEY_COMPLEX,1, (0, 255, 0), 2) 
		# If any hand present
        else:
            for i in result.multi_handedness: 
				
				# Return whether it is Right or Lef
                label = MessageToDict(i)['classification'][0]['label']
                if label == 'Left': 	
					# Display 'Left Hand' on left side of window
                    cv.putText(image, label+' Hand',(20, 50),cv.FONT_HERSHEY_COMPLEX,1,  2)
                if label == 'Right': 
					# Display 'Left Hand' on left side of window
                    cv.putText(image, label+' Hand', (460, 50),cv.FONT_HERSHEY_COMPLEX,1,  2) 

    
    cv.imshow('Hand Tracker', image)
    if cv.waitKey(1)==13:                
        break
    
