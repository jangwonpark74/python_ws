import socket

target_host = "127.0.0.1"
target_port = 5005 

# create a socket object
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server.bind((target_host, target_port))

while True:
    data, addr = server.recvfrom(4096) # buffer size is 4096
    print("received msg : ", data.decode())
