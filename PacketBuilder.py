import struct


class PacketBuilder:
    @staticmethod
    def build_packet(url, query_type):
        # reference: https://stackoverflow.com/questions/24814044/having-trouble-building-a-dns-packet-in-python

        pack = struct.pack('>H', 12049)  # Query Ids (Just 1 for now)
        pack += struct.pack('>H', 256)  # Flags
        pack += struct.pack('>H', 1)  # Questions
        pack += struct.pack('>H', 0)  # Answers
        pack += struct.pack('>H', 0)  # Authorities
        pack += struct.pack('>H', 0)  # Additional
        for part in url.split('.'):
            pack += struct.pack('B', len(part))
            for byte in bytes(part):
                pack += struct.pack('c', byte.encode('utf-8'))
        pack += struct.pack('B', 0)  # End of String

        # reference for query type code: https://www.iana.org/assignments/dns-parameters/dns-parameters.xhtml
        q_type = 0
        if query_type == 'MX':
            q_type = 15
        elif query_type == 'NS':
            q_type = 2
        else:
            q_type = 1

        pack += struct.pack('>H', q_type)  # Query Type
        pack += struct.pack('>H', 1)  # Query Class
        return pack
