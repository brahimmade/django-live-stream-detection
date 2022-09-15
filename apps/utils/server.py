import socket
import threading
import cv2
import pickle
import imutils
import time
import struct
import sys
from collections import Counter
from ast import literal_eval
# import torch
import settings_server
import processvideo

model = processvideo.load_ml_model()
# model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using Device: ", model.device)


class ThreadedServer(object):
    def __init__(self, host, port, number, vidname=None, stream=None):
        self.vidname = vidname
        self.stream = stream
        self.host = host
        self.port = port
        self.number = number

        # self.fps = 1/60
        # self.fps_ms = int(self.fps*100)
        # print("self.fps_ms: ", self.fps_ms)

        self.prepare_socket()

    def prepare_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("[SUCCESS] Socket created.")
        try:
            self.address = (self.host, self.port)
            self.sock.bind(self.address)
            print("[SUCCESS] Socket binding done.")
        except socket.error as msg:
            print('Bind failed. Error : ' + str(sys.exc_info()))
            sys.exit()

    def listen(self):
        self.sock.listen(5)
        print("[SUCCESS] Server listening at: ", self.address)
        while True:
            client, address = self.sock.accept()
            print('GOT CONNECTION FROM:', address)
            client.settimeout(settings_server.CLIENT_TIMEOUT)
            threading.Thread(target=self.listenToClient,
                             args=(client, address)).start()
            print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

    def listenToClient(self, client, address):
        while True:
            try:
                if client:
                    vidfl = processvideo.VideoLoader()
                    if self.vidname:
                        vid = vidfl.load_local_vid(self.vidname)
                    elif self.stream:
                        vid = vidfl.load_stream(self.stream)
                    else:
                        vid = vidfl.load_webcam()
                    data_dict = {}
                    while (vid.isOpened()):
                        img, frame = vid.read()
                        time.sleep(0.4)
                        frame = imutils.resize(
                            frame, width=settings_server.FRAME_WIDTH)
                        # # Inference code starts
                        # # perform inference
                        all_classes = model.names

                        results = model(frame)
                        # labels, cord = results.xyxyn[0][:, -
                        #                                 1], results.xyxyn[0][:, :-1]
                        # parse results
                        # predictions = results.pred[0]
                        # boxes = predictions[:, :4]
                        # scores = predictions[:, 4]
                        # categories = predictions[:, 5]

                        # # show detection bounding box on image
                        # results.show()
                        results_list = results.pandas(
                        ).xyxy[0].to_json(orient="records")
                        results_list = literal_eval(results_list)
                        classes_list = [item["name"] for item in results_list]
                        results_counter = Counter(classes_list)
                        # print(results_counter)
                        results.render()

                        # # save results into results folder
                        # results.save(save_dir='results/')
                        # # inference code ends

                        data_dict['frame'] = frame
                        data_dict['detection_info'] = results_counter
                        data_dict['all_classes'] = all_classes
                        data_dict['results_list'] = results_list
                        a = pickle.dumps(data_dict)
                        message = struct.pack("Q", len(a))+a
                        time.sleep(0.5)
                        try:
                            # send message or data frames to client
                            client.sendall(message)
                        except Exception as e:
                            print(e)
                            raise Exception(e)
                        if settings_server.VIEW_TRANSMITTING:
                            cv2.imshow(
                                f'TRANSMITTING VIDEO {self.number}', frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                client.close()
                else:
                    print('Client disconnected')
            except:
                client.close()
                print("some error occured")
                return False


def main():
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print('HOST IP:', host_ip)
    while True:
        port_num = settings_server.PORT1
        try:
            port_num = int(port_num)
            break
        except ValueError as e:
            print(e)
            pass

    # detect camera
    # ThreadedServer(host_ip, port_num, 1).listen()
    # detect local video file (place the videofile in directory: config/datainput/videos)
    ThreadedServer(host_ip, port_num, 1, 'shortvid.avi').listen()
    # detect stream
    # ThreadedServer(host_ip, port_num, 1, streampath-here).listen()


def run():
    process_1 = threading.Thread(target=main, name="first_inference").start()
    # process_2 = threading.Thread(
    #     target=main_2, name="second_inference").start()


if __name__ == "__main__":
    run()
