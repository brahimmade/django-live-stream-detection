import socket
import cv2
import pickle
import struct
import settings_client

# create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = settings_client.HOST_IP
port = settings_client.PORT1
address = (host_ip, port)
client_socket.connect(address)
data = b""
payload_size = struct.calcsize("Q")

while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)  # 4K
        if not packet:
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    data_dict = pickle.loads(frame_data)
    frame = data_dict['frame']
    detection_info = str(dict(data_dict['detection_info']))
    print(detection_info)
    if settings_client.WRITE_IMAGE_INFO:
        cv2.putText(frame,
                    detection_info,
                    (25, 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=settings_client.FONT_SCALE,
                    color=settings_client.FONT_COLOR,
                    thickness=settings_client.FONT_THICHNESS,
                    lineType=cv2.LINE_AA
                    )
    cv2.imshow("RECEIVING VIDEO", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # press q to exit video
        break
client_socket.close()
