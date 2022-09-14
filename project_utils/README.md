# Stream video after object detection (Socket-Client)

### Objectives
- Object detection using pretrained or custom ML model
- Stream video to multiple clients using python sockets and threading (multiple clients for a single video-detection)
- Object detection on multiple streams on separate threads (object-detection of the multiple streams)

## ML object detection algorith used
- YoloV5

## For streaming videos
- python sockets

## For videos reading, frames extraction, frame reading and display
- opencv

### Procedure
- Create a virtual environment
```bash
virtualenv venv

# activate environment
# for linux
source venv/bin/activate

# for windows
venv\Scripts\activate

# install yolov5
pip install yolov5
```
### Place weight file
- place weight file in config/weights/

### Place video file to be detected at
- If detection is for the local video file, place the file ar config/datainput/videos

### Edit server_settings.py
- edit the ip address, port and other required settings

### Start server.py

```python
python server.py
```

### Start client.py
In a new terminal
```python
# open new terminal and run
python client.py
```