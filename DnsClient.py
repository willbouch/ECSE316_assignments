import argparse
from Client import DnsClient


def main():

    parsed_args = parse()

    name = parsed_args.name
    server = parsed_args.server
    port = parsed_args.port
    retries = parsed_args.retries
    timeout = parsed_args.timeout
    mx = parsed_args.mx
    ns = parsed_args.ns
    dnsClient = DnsClient(name=name, server=server,
                          timeout=timeout, retries=retries, port=port, mx=mx, ns=ns)

    dnsClient.send_request()


def parse():
    # reference: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    mut_exc_g = parser.add_mutually_exclusive_group(required=False)
    mut_exc_g.add_argument('-mx', action='store_true',
                           default=False, dest='mx', help='MX type')
    mut_exc_g.add_argument('-ns', action='store_true',
                           default=False, dest='ns', help='NS type')
    parser.add_argument('-t', help='timeout', required=False,
                        type=int, action='store', dest='timeout', default=5)
    parser.add_argument('-r', help='max-retries', required=False,
                        type=int, action='store', dest='retries', default=3)
    parser.add_argument('-p', help='port', required=False,
                        type=int, action='store', dest='port', default=53)

    parser.add_argument(help='IPv4 address',
                        action='store', dest='server')
    parser.add_argument(help='domain name',
                        action='store', dest='name')
    return parser.parse_args()


if __name__ == '__main__':
    main()