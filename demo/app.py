
import argparse
import json
from bottle import get, post, run, request, response, template

from keras.models import load_model
from lasagne_bilstm import BiRNN
from casket.nlp_utils import Indexer

from ..predict import CharFiller

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model prefix')
    parser.add_argument('-t', '--type', required=True,
                        help='model type (keras/lasagne)')

    args = parser.parse_args()

    print("Loading indexer")
    idxr = Indexer.load(args.model + '_indexer.json')

    print("Loading model")
    if args.type == 'keras':
        model = load_model(args.model + '.h5')
    elif args.type == 'lasagne':
        model = BiRNN.load(args.model)
    else:
        raise ValueError("Didn't understand model [%s]" % args.model)

    print("Done")
    filler = CharFiller(model, idxr, context=10)

    def render_template(text=None, output=None):
        return template('demo/landing', text=text, output=output)

    def get_prediction(pred, with_prob):
        if with_prob:
            return pred[0][0]
        else:
            return pred[0]

    def normalized_pred(text, pos, max_n, with_prob):
        pred = filler.predict(text, pos, max_n=max_n, with_prob=with_prob)
        if with_prob:
            return [(c, float(prob)) for (c, prob) in pred]
        else:
            return pred

    @get('/')
    def landing():
        return render_template()

    @post('/fillrest')
    def fillrest():
        response.headers['Content-Type'] = 'application/json'
        try:
            text = request.POST.text.strip()
            max_n = int(request.POST.maxN) or 1
            with_prob = request.POST.withProb or False
            fillers = {pos: normalized_pred(text, pos, max_n, with_prob)
                       for (pos, c) in enumerate(text) if c == '_'}
            output = "".join([get_prediction(fillers[pos], with_prob)
                              if pos in fillers else c
                              for (pos, c) in enumerate(text)])
            response.code = 200
            return json.dumps({'output': output, 'fillers': fillers})
        except Exception as e:
            response.code = 500
            return json.dumps({'error': str(e)})

    @post('/fill')
    def fill():
        text = request.POST.text.strip()
        output = "".join([filler.predict(text, pos, with_prob=False)[0]
                          if c == "_" else c
                          for (pos, c) in enumerate(text)])
        return render_template(text=text, output=output)

    run(host='0.0.0.0', port=8083)
