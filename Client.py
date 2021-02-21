import socket as skt
from PacketBuilder import PacketBuilder
import time


class DnsClient:
    def __init__(self, name, server, timeout, retries, port, mx, ns):
        self.name = name
        self.server = server[1:]  # Remove the @
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
        packet = PacketBuilder().build_packet(
            url=self.name, query_type=self.query_type)
        socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
        socket.settimeout(self.timeout)
        socket.setsockopt(skt.SOL_SOCKET, skt.SO_REUSEADDR, 1)
        socket.bind(('', self.port))

        # Send and receive
        resp = self.send_and_receive(socket, packet, 1)

        # Interpret
        return PacketBuilder().unbuild_packet(resp)

    def send_and_receive(self, socket, packet, i):
        if i - 1 <= self.retries:
            try:
                t0 = time.time()

                # Send
                socket.sendto(packet, (self.server, self.port))

                # Receive
                resp, _ = socket.recvfrom(512)

                t1 = time.time()

                # Close
                socket.close()

                print('Response received after ' + str(t1-t0) +
                      ' seconds ('+str(i-1)+' retries)')

                return resp
            except skt.timeout:
                print('ERROR\tSocket timeout on attempt ' + str(i))
                self.send_and_receive(socket, packet, i+1)
            except socket.error as e:
                print('ERROR\tCould not create socket')
            except socket.gaierror as e:
                print('ERROR\tUnknown host')
            except Exception as e:
                print(e)

        elif i - 1 > self.retries:
            print('ERROR\tMaximum number of retries ' +
                  str(self.retries)+' exceeded')
            return -1
