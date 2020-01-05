import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

i = 0
while True:
    i = i + 1
    #  Do some 'work'
    time.sleep(5)
    print("send message")
    #  Send reply back to client
    socket.send_json(
        {
            "name":"World" + str(i),
            "emotion": "happy"
        }
    )
