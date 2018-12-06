#!/usr/bin/env python
# from gevent import monkey; monkey.patch_all()
import argparse
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from bert.extract_features import PoolingStrategy
from service.client import BertClient
from service.server import BertServer

STATUS_OK = "ok"
STATUS_ERROR = "error"


def start(args,
          url_root="./bert",
          host="0.0.0.0",
          port=8080):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    app.config['JSON_AS_ASCII'] = False
    server = BertServer(args)
    server.start()

    bert_client = BertClient(port=args.port_in, port_out=args.port_out, output_fmt='list')

    @app.route('/', methods=['GET'])
    def index():
        return jsonify("hello bert")

    @app.route('/encode', methods=['POST'])
    def encode():
        inputs = request.get_json(force=True)
        out = {}
        try:
            data = bert_client.encode(inputs)
            out['status'] = STATUS_OK
            out['error'] = ''
            out['data'] = data
        except Exception as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)

    CORS(app)

    http_server = WSGIServer((host, port), app)
    print("Model loaded, serving bert on port %d" % port)
    http_server.serve_forever()
    server.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", type=str, default="0.0.0.0")
    parser.add_argument("-port", type=int, default="8080")
    parser.add_argument("-url_root", type=str, default="/bert")
    parser.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-max_seq_len', type=int, default=25,
                        help='maximum length of a sequence')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    parser.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    parser.add_argument('-port_in', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for outputting result to client')
    parser.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. '
                             'Give a list in order to concatenate several layers into 1.')
    parser.add_argument('-pooling_strategy', type=PoolingStrategy.from_string,
                        default=PoolingStrategy.REDUCE_MEAN, choices=list(PoolingStrategy),
                        help='the pooling strategy for generating encoding vectors')
    parser.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determines the fraction of the overall amount of memory '
                             'that each visible GPU should be allocated per worker. '
                             'Should be in range [0.0, 1.0]')
    args = parser.parse_args()
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


if __name__ == '__main__':
    args = get_args()
    start(args, url_root=args.url_root, host=args.ip, port=args.port)
