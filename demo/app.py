
from bottle import get, post, run, request, template

from ..predict import load_emb_bilstm, CharFiller
from ..corpus import Indexer

print("Loading model")
idxr = Indexer.load('./fitted/emb_bilstm_indexer.json')  # path from project root
m = load_emb_bilstm(idxr.vocab_len(), 28, lstm_layer=250, hidden_layer=150, rnn_layers=3)
filler = CharFiller(m, idxr, 10)
print("Done")


def render_template(text=None, output=None):
    return template('demo/landing', text=text, output=output)


@get('/')
def landing():
    return render_template()


@post('/fill')
def fill():
    text = request.POST.text.strip()
    output = "".join([filler.predict(text, pos) if c == '_' else c
                      for (pos, c) in enumerate(text)])
    return render_template(text=text, output=output)

if __name__ == '__main__':
    run(host='localhost', port=8083)

