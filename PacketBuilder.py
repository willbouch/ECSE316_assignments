import struct


class PacketBuilder:
    def build_packet(self, url, query_type):
        # reference: https://stackoverflow.com/questions/24814044/having-trouble-building-a-dns-packet-in-python

        pack = struct.pack('>H', 12049)  # Query Ids (Just 1 for now) TODO
        pack += struct.pack('>H', 0x0100)  # Flags
        pack += struct.pack('>H', 1)  # Questions
        pack += struct.pack('>H', 0)  # Answers
        pack += struct.pack('>H', 0)  # Authorities
        pack += struct.pack('>H', 0)  # Additional
        for part in url.split('.'):
            pack += struct.pack('B', len(part))
            for byte in part:
                pack += struct.pack('c', byte.encode('utf-8'))
        pack += struct.pack('B', 0)  # End of String

        # reference for query type code: https://www.iana.org/assignments/dns-parameters/dns-parameters.xhtml
        q_type = 0
        if query_type == 'MX':
            q_type = 0x000F
        elif query_type == 'NS':
            q_type = 0x0002
        else:
            q_type = 0x0001

        pack += struct.pack('>H', q_type)  # Query Type
        pack += struct.pack('>H', 0x0001)  # Query Class
        return pack

    def unbuild_packet(self, msg):
        # reference: https://stackoverflow.com/questions/16977588/reading-dns-packets-in-python

        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |                     ID                        |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |QR|   Opcode    |AA|TC|RD|RA|   Z   |   RCODE  |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |                   QDCOUNT                     |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |                   ANCOUNT                     |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |                   NSCOUNT                     |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        # |                   ARCOUNT                     |
        # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

        id, flags, qdcount, ancount, _, arcount = struct.Struct(
            '!6H').unpack_from(msg)
        # We start offset after the Header section
        offset = struct.Struct('!6H').size

        aa = (flags & 0x0400) != 0
        # We don't use the other bits for outputting purposes on this lab

        # Skip questions section
        for _ in range(qdcount):
            _, offset = self.decode_labels(msg, offset)  # Skip QNAME
            offset += struct.Struct('!2H').size  # Skip QTYPE and QCLASS

        # Answer section
        answers, offset = self.get_answer_section(msg, offset, aa, ancount)

        # Additional section
        additional, offset = self.get_answer_section(msg, offset, aa, arcount)

        return {
            'aa': aa,
            'answer_count': ancount,
            'additional_info_count': arcount,
            'id': id,
            'query_count': qdcount,
            'answers': answers,
            'additional': additional
        }

    def get_answer_section(self, msg, offset, aa, count):
        answers = []
        auth = 'auth' if aa else 'nonauth'
        # Answer section
        for _ in range(count):
            _, offset = self.decode_labels(msg, offset)
            an_type = struct.Struct('!H').unpack_from(msg, offset)[0]
            offset += struct.Struct('!2H').size  # Skip CLASS
            ttl = struct.Struct('!I').unpack_from(msg, offset)[0]
            offset += struct.Struct('!I').size
            offset += struct.Struct('!H').size  # SKIP RDLENGTH

            an_rdata = None
            if an_type == 0x0001:  # Type A
                fi, s, t, fo = struct.Struct('!4B').unpack_from(msg, offset)
                offset += struct.Struct("!4B").size
                ip = str(fi)+'.'+str(s)+'.'+str(t)+'.'+str(fo)
                an_rdata = 'IP\t'+ip+'\t'+str(ttl)+'\t'+auth

            elif an_type == 0x0002:  # Type NS
                name, offset = self.decode_labels(msg, offset)
                an_rdata = 'NS\t' + \
                    self.process_labels(name)+'\t'+str(ttl)+'\t'+auth

            elif an_type == 0x0005:  # Type CNAME
                name, offset = self.decode_labels(msg, offset)
                an_rdata = 'CNAME\t' + \
                    self.process_labels(name)+'\t'+str(ttl)+'\t'+auth

            elif an_type == 0x000F:  # Type MX
                preference = struct.Struct('!H').unpack_from(msg, offset)[0]
                offset += struct.Struct('!H').size
                exchange, offset = self.decode_labels(
                    msg, offset)  # Skip Exchange
                an_rdata = 'MX\t' + \
                    self.process_labels(exchange)+'\t' + \
                    str(preference)+'\t'+str(ttl)+'\t'+auth

            answers.append(an_rdata)
        return answers, offset

    def decode_labels(self, message, offset):
        labels = []

        while True:
            length, = struct.unpack_from('!B', message, offset)

            if (length & 0xC0) == 0xC0:
                pointer, = struct.unpack_from('!H', message, offset)
                offset += 2

                return (list(labels) + list(self.decode_labels(message, pointer & 0x3FFF))), offset

            if (length & 0xC0) != 0x00:
                raise Exception('unknown label encoding')

            offset += 1

            if length == 0:
                return labels, offset

            labels.append(*struct.unpack_from('!%ds' %
                                              length, message, offset))
            offset += length

    def process_labels(self, labels):
        def flatten(lst):
            if not isinstance(lst, list):
                return [lst]

            res = []
            for el in lst:
                if isinstance(el, list):
                    res += flatten(el)
                elif not isinstance(el, int):
                    res.append(el)

            return res

        flat_list = flatten(labels)
        flat_list = b'.'.join(flat_list)
        return flat_list.decode()
