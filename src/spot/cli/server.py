def add_subparser(parent):
    parser = parent.add_parser('server', help='Serve a Faster R-CNN network')
    parser.set_defaults(func=run)

    parser.add_argument(
            'model', metavar='MODEL',
            help='model factory',
            type=file)

    parser.add_argument(
            'weights', metavar='WEIGHTS',
            help='model pre-trained weights',
            type=file)

    parser.add_argument(
            '-p', '--port', dest='port',
            help='port to listen on',
            default=80, type=int)

def run(args):
    from flask import Flask
    app = Flask(__name__)

    net = caffe.Net(args.model.name, args.weights.name, caffe.TEST)

    # TODO: Hook up to prediction infra
    @app.route('/predict', methods=['POST'])
    def predict():
        if len(flask.request.files) > 0:
            # grab files
            files = flask.request.files.items()
        else:
            im = cv2.imread(imdb.tags[i]['image'])
            scores, boxes = test.im_detect(net, im, box_proposals)
            url = flask.request.args['url']
            files = [StringIO(urllib.urlopen(url).read())]

        image = caffe.io.load_image(files)

    app.run(port=args.port)

    # net, imdb, max_per_image=100, thresh=0.05, vis=False):
    # im = cv2.imread(imdb.tags[i]['image'])
    # scores, boxes = im_detect(net, im)

    # # skip j = 0, because it's the background class
    # for j in xrange(1, imdb.num_classes):
    #     inds = np.where(scores[:, j] > thresh)[0]
    #     cls_scores = scores[inds, j]
    #     cls_boxes = boxes[inds, j*4:(j+1)*4]
    #     cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
    #         .astype(np.float32, copy=False)
    #     keep = nms(cls_dets, cfg.TEST.NMS)
    #     # all detections are collected into:
    #     #    all_boxes[cls][image] = N x 5 array of detections in
    #     #    (x1, y1, x2, y2, score)
    #     cls_dets = cls_dets[keep, :]
    #     label = imdb.classes[j]
