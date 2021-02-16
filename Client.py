import socket as skt
from PacketBuilder import PacketBuilder
import time


class DnsClient:
    def __init__(self, name, server, timeout, retries, port, mx, ns):
        self.name = name
        self.server = server
        self.timeout = timeout
        self.retries = retries
        self.port = port

        if mx:
            self.query_type = 'MX'
        elif ns:
            self.query_type = 'NS'
        else:
            self.query_type = 'A'

    def send_request(self):
        print('DnsClient sending request for ' + self.name)
        print('Server: ' + self.server)
        print('Request type: ' + self.query_type)

        # Used https://wiki.python.org/moin/UdpCommunication for information about the socket library
        packet = PacketBuilder.build_packet(
            url=self.name, query_type=self.query_type)
        socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
        socket.settimeout(self.timeout)
        socket.bind(('', self.port))

        self.send_and_receive(socket, packet)

    def send_and_receive(self, socket, packet):
        is_success = False
        i = 1
        while (not is_success) and i - 1 <= self.retries:
            try:

                t0 = time.time()

                # Send
                socket.sendto(packet, (self.server, self.port))
                is_success = True

                # Receive
                resp, addr = socket.recvfrom(512)

                t1 = time.time()

                # Close
                socket.close()

                print('Response received after ' + (t1-t0) +
                      ' seconds ('+(i-1)+' retries)')
            except skt.timeout:
                print('ERROR: Socket timeout on attempt ' + i)

        if i - 1 > self.retries:
            print('ERROR: Maximum number of retries ('+self.retries+') exceeded')
            return
