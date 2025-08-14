import tensorflow as tf 
import numpy as np 
import cv2 

choice = int(input("choice Cam [0] for intern cam and [1] external cam : "))
cap = cv2.VideoCapture(choice)


    

interpreter = tf.lite.Interpreter(model_path="Quantile_Model2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def inference_models (inputs_data) :
    interpreter.set_tensor(input_details[0]['index'],inputs_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.argmax(output_data,axis=-1)
    return output_data
data_list = {
    0 : 'apple' ,
    1 : 'banana' ,
    2: 'beetroot' ,
     3 : 'bell peper' ,
    4 : 'cabbage' ,
     5 :'capsicum' ,
    6 : 'carrot' ,
    7 : 'cauliflower' ,
    8 : 'carrot' ,
    9 : 'cauliflower' ,
    10 : 'chilli pepper' ,
    11 :'cucumbar' ,
    12 : 'eggplant' ,
    13 : 'garlic' ,
    14: 'ginger' ,
    15 :'grapes' ,
    16 : 'jalepeno' ,
    17 : 'kiwi' ,
    18 :'lemon' ,
    19 :'lettucu' ,
    20 : 'mango' ,
    21 : 'union' ,
    22 : 'orange',
    23 :'paprika',
    24 : 'pear' ,
    25 :'peas' ,
    26 : 'pineple',
    27 :'pomegranate',
    28 : 'potato' ,
    29 : 'raddish',
    30 :'soybeand',
    31 :'spinach' ,
    32 :'sweetcorn',
    33 :'sweetpotato',
    34 :'tomato',
    35 :'turnip',
    36 :'watermelon'

}

while True :
    ret,frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame,(244,244))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    
    if not ret :
        break 
    
    outputs = inference_models(img)
    name = data_list.keys()
    text = f"Prediction: {data_list.get(outputs[0])}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cap.destroyAllWindow()
