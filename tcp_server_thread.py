import socket
import threading

'''
What is the Difference Between 127.0.0.1 and 0.0.0.0?
127.0.0.1 is the loopback address (also known as localhost).
0.0.0.0 is a non-routable meta-address used to designate 
an invalid, unknown, or non-applicable target (a ‘no particular address’ place holder).
In the context of a route entry, it usually means the default route.
In the context of servers, 0.0.0.0 means all IPv4 addresses on the local machine.
If a host has two IP addresses, 192.168.1.1 and 10.1.2.1, 
and a server running on the host listens on 0.0.0.0,
it will be reachable at both of those IPs.

What is the IP Address 127.0.0.1?
127.0.0.1 is the loopback Internet protocol (IP) address also referred to as the localhost.

The address is used to establish an IP connection to the same machine or
computer being used by the end-user.
'''

bind_ip = "0.0.0.0"
bind_port = 9999

server = server.socket(socket.AF_INET, socket.SOCK_STREAM)

# To start off TCP server pass the IP address and port we want listen on. 
server.bind((bind_ip, bind_port))

# set the maximum backlog of connections set to 5
server.listen(5)

print("[*] Listening on %s:%d" % (bind_ip, bind_port))

# this is our client-handling thread
def handle_client(client_socket):

    #print out what the client sends
    request = client_socket.recv(4096)

    print("[*] Received: %s" % request.decode())

    client_socket.send("ACK!")
    client_socket.close()


while True:
    client, addr = server.accept()

    print("[*] Accepted connection from : %s:%d" % (addr[0], addr[1]))

    client_handler = threading.Thread(target=handle_client, args=(client,)  )
    client_handler.start()
