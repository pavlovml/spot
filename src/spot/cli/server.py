import json, urllib, os.path, spot.test
import numpy as np
from spot.nms import nms
from spot.config import cfg

def add_subparser(parent):
    parser = parent.add_parser('server', help='Serve a Faster R-CNN network')
    parser.set_defaults(func=run)

    parser.add_argument(
            'model', metavar='MODEL',
            help='model directory',
            type=str)

    parser.add_argument(
            'weights', metavar='WEIGHTS',
            help='model pre-trained weights',
            type=file)

    parser.add_argument(
            '-p', '--port', dest='port',
            help='port to listen on',
            default=80, type=int)

def download_image(url):
    return read_image(urllib.urlopen(url))

def read_image(io):
    import cv2
    array = np.fromstring(io.read(), np.uint8)
    return cv2.imdecode(array, cv2.CV_LOAD_IMAGE_COLOR)

def run(args):
    import caffe, flask

    app = flask.Flask(__name__)

    labels_filename = os.path.join(args.model, 'labels.txt')
    test_filename = os.path.join(args.model, 'test.prototxt')
    net = caffe.Net(test_filename, args.weights.name, caffe.TEST)

    with open(labels_filename, 'r') as f:
        labels = f.read().splitlines()

    @app.route('/predict', methods=['POST'])
    def predict():
        files = flask.request.files
        if len(files) > 0:
            images = [read_image(f) for _, f in files.iteritems()]
        else:
            url = flask.request.args['url']
            images = [download_image(url)]

        thresh = float(flask.request.args.get('thresh', 0.05))

        tags = []
        for image in images:
            scores, boxes = spot.test.im_detect(net, image)

            # skip j = 0, because it's the background class
            for j in xrange(1, len(labels)):
                inds = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                keep = nms(cls_dets, cfg.TEST.NMS)
                # cls_dets is an array of (x1, y1, x2, y2, score)
                cls_dets = cls_dets[keep, :]
                label = labels[j]
                for row in cls_dets:
                    tags.append({
                        'label': label,
                        'score': float(row[4]),
                        'bounds': [[float(row[0]), float(row[1])], [float(row[2]), float(row[3])]]
                    })

        return json.dumps(tags)

    app.run(host='0.0.0.0', port=args.port, debug=True)
