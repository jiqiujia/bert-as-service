# using BertClient in sync way

import sys
import time

from service.client import BertClient

if __name__ == '__main__':
    bc = BertClient(ip=sys.argv[1], port=int(sys.argv[2]), port_out=int(sys.argv[3]))
    # encode a list of strings
    with open('README.md') as fp:
        data = [v for v in fp if v.strip()]
    print(data[0])
    for j in range(1, 200, 10):
        start_t = time.time()
        tmp = data * j
        print(tmp)
        bc.encode(tmp)
        time_t = time.time() - start_t
        print('encoding %d strs in %.2fs, speed: %d/s' % (len(tmp), time_t, int(len(tmp) / time_t)))
