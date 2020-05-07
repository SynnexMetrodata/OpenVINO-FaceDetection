# build for OpenVINO workshop
# running using OpenVINO toolkit 2020 Release 1

from openvino.inference_engine import IECore, IENetwork
import cv2

if __name__ == "__main__":
    # initialize object
    ie = IECore()
    net = IENetwork(model="face-detection-adas-0001.xml", weights="face-detection-adas-0001.bin")
    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)

    # load camera source
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            blob = cv2.dnn.blobFromImage(frame, size=(672, 384))
            output = exec_net.infer(inputs={'data': blob})
            
            # process inference result
            for face in next(iter(next(iter(output['detection_out'])))):
                image_id = face[0]
                lavel = face[1]
                conf = face[2]
               
                # count the coordinate on frame
                (x1, y1) = (int(face[3]*frame.shape[1]), int(face[4]*frame.shape[0]))
                (x2, y2) = (int(face[5]*frame.shape[1]), int(face[6]*frame.shape[0]))

                # draw rectangle with confidence level > 0.5
                if conf > 0.5:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            # display frame
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
                