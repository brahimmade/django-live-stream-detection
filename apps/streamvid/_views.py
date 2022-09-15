import socket
import cv2
import pickle
import struct
from utils import settings_client
import pybboxes as pbx

from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators import gzip


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


def get_video_stream():
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
        all_classes = data_dict['all_classes']
        predictions = data_dict['predictions']
        boxes = data_dict['boxes']
        if settings_client.DETECT_OBJECT:
            dh, dw, _ = frame.shape
            if predictions:
                for *xywh, conf, cls in predictions:
                    # print(conf)
                    # print(cls)
                    # for bbox in boxes:
                    x1, y1, x2, y2 = xywh
                    = int(
                        row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    x, y, w, h = xywh
                    bbox_left = x
                    bbox_top = y
                    bbox_w = w - x
                    bbox_h = h - y
                    print(bbox_left, bbox_top, bbox_w, bbox_h)
                    cv2.rectangle(
                        frame,
                        (bbox_left, bbox_top),
                        (bbox_w, bbox_h),
                        # color_list[int(category_id)],
                        (100, 0, 100),
                        1,
                    )

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
        ret, buf = cv2.imencode(".jpg", frame)
        frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield all_classes


@gzip.gzip_page
def video(request):
    """
    This is the heart of our video display. Notice we set the mimetype to 
    multipart/x-mixed-replace. This tells Flask to replace any old images with 
    new values streaming through the pipeline.
    """
    try:
        return StreamingHttpResponse(
            get_video_stream(),
            content_type='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        print(e)

    return render(request, 'index.html')
    # <img src="{{ url_for('video_feed') }}" height="80%">
