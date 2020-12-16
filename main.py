import cv2
import time
import numpy as np
import argparse
import threading
import json

#416x416
def parse_arguments():
    parser = argparse.ArgumentParser(prog ='main')
    parser.add_argument('-m','--mode',type=int,help='mode type 0->img,1->video,2->live',required=True)
    parser.add_argument('-fp','--filepath',type=str,help='file path containing camera ips or video/img files paths',required=True)
    parser.add_argument('-wp','--weightsfile',type=str,help='file path containing trained weights',required=True)
    parser.add_argument('-cfg','--cfgfile',type=str,help='file path containing model cfg',required=True)
    parser.add_argument('-n','--namesfile',type=str,help='file path containing class names',required=True)
    args = parser.parse_args()
    
    return args

def parse_weights(path):
    weightspath = []
    with open(path,'r') as f:
        weightspath = [i.strip("\n") for i in f.readlines()]
    f.close()
    return weightspath

def parse_filepaths(path):
    filepaths=[]
    with open(path,'r') as f:
        filepaths = [i.strip("\n") for i in f.readlines()]
    f.close()
    return filepaths

def parse_namesfile(path):
    classes =[]
    with open(path,'r') as f:
        classes = [i.strip("\n") for i in f.readlines()]
    f.close()
    return classes

def predictions(frame,predictor,layers,size):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, size, swapRB=True, crop=False)
    predictions =None      
    predictor.setInput(blob)
    start = time.time()
    predictions = predictor.forward(layers)
    end = time.time()
    print("prediction time: {:.2f} seconds".format(end - start))
    return predictions

def filter_predictions(predictions,classnames,pthres,nmsthres,width,height):
    boxes = []
    confidences =[]
    classes =[]
    for prediction in predictions:
        #loop over each layer output
        for output in prediction:
            scores = output[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            #discard lesser confident bounding boxes
            if confidence > pthres:
                classes.append(classnames[classid])
                confidences.append(float(confidence))
                box = output[0:4] *np.array([width,height,width,height])
                center_X,center_Y,b_width,b_height = box.astype("int")
                x = int(center_X -(b_width/2))
                y = int(center_Y - (b_height/2))
                # #scaling width and height of bounding box
                # width = int(output[2] *self.width)
                # height = int(output[3]*self.height)

                # #locating starting point of box       # taken help from https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
                # center_X = int(output[0]* self.width)
                # center_Y = int(output[1]*self.height)
                # start_X = abs(int(center_X - self.width/2))
                # start_Y = abs(int(center_Y- self.height/2))
                boxes.append([x,y,int(b_width),int(b_height)])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, pthres, nmsthres)
    return (classes,confidences,boxes,indices)

def draw_boxes(cords,frame,name,files):
    global sync_record
    temp = []
    classes,confidences,boxes,indices = cords
    if len(indices) < 1:
        return frame
    for i in indices.flatten():
        color = ()
        if classes[i] == 'standing':
            color = (255,0,0)
        else:
            color = (0,0,255)
        if sync_record:
            cv2.putText(frame,'(REC)',(50,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
            temp.append({
                'start_X':boxes[i][0],
                'start_Y':boxes[i][1],
                'width'  : boxes[i][0] + boxes[i][2],
                'height' : boxes[i][1] + boxes[i][3],
                'class'  : classes[i]
            })
        cv2.rectangle(frame,(boxes[i][0],boxes[i][1]),(boxes[i][0] + boxes[i][2],boxes[i][1] + boxes[i][3]),color,2)
        text = "{}: {:.2f}".format(classes[i], confidences[i])
        cv2.putText(frame, text, (boxes[i][0], boxes[i][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    #cv2.imwrite("example.png", frame)
    files[name].append(temp)
    return frame

def capture(camera,predictor,layers,classes,p_thres,nms_thres,name,files):
    global sync_switch
    global sync_record
    frames =0
    while True:
        if sync_switch:
            break
        ret,frame = camera.read()
        if not ret:
            break
        #frame,predictor,layers,size
        frames+=1
        preds = predictions(frame,predictor,layers,(416,416))
        frame = cv2.resize(frame,(600,300))
        preds = filter_predictions(preds,classes,p_thres,nms_thres,frame.shape[1],frame.shape[0])
        frame = draw_boxes(preds,frame,name,files)
        cv2.imshow(name,frame)
        if name == 'camera0':
            cv2.moveWindow(name,0,0)
        elif name == 'camera1':
            cv2.moveWindow(name,0,600)
        else:
            cv2.moveWindow(name,800,0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            sync_switch = True
            break
        if cv2.waitKey(33) & 0xFF == ord("a"):
            sync_record = True

    camera.release()
    
if __name__ == "__main__":
    sync_switch = False
    sync_record = True
    p = 0.1
    nms = 0.9
    args = parse_arguments()
    mode,filepath,weightsfile,cfgfile,namesfile = args.mode,args.filepath,args.weightsfile,args.cfgfile,args.namesfile
    #weights = parse_weights(weightsfile)
    classes = parse_namesfile(namesfile)
    cameras =[]
    predictors=[]
    out_layers = []
    if mode == 1:
        movie_files = parse_filepaths(filepath)
        for i,movie in enumerate(movie_files):
            cameras.append(cv2.VideoCapture(movie))
            predictors.append(cv2.dnn.readNetFromDarknet(cfgfile,weightsfile))
        t_layers = predictors[0].getLayerNames()
        out_layers = [t_layers[i[0] - 1] for i in predictors[0].getUnconnectedOutLayers()]
    elif mode == 2:
        ips = parse_filepaths(filepath)
        for i,ip in enumerate(ips):
            cameras.append(cv2.VideoCapture(ip))
            predictors.append(cv2.dnn.readNetFromDarknet(cfgfile,weightsfile))
        t_layers = predictors[0].getLayerNames()
        out_layers = [t_layers[i[0] - 1] for i in predictors[0].getUnconnectedOutLayers()]
    else:
        exit()
    threads = []
    filehandlers = {}
    for i,camera in enumerate(cameras):
        filehandlers['camera' + str(i)] = []   #open('camera' + str{i}+'txt','w')
        t1 = threading.Thread(target=capture,args=(camera,predictors[i],out_layers,classes,p,nms,'camera' +str(i),filehandlers),daemon = True)
        threads.append(t1)
        t1.start()
    for i in threads:
        i.join()
    cv2.destroyAllWindows()
    for f in filehandlers.keys():
        with open('boxes/'+f+'.txt','w') as fi:
            json.dump(filehandlers[f],fi)
        fi.close()
    
        

    














    


    