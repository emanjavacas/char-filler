# coding: utf-8

from unsmoothed_lm import UnsmoothedLM


if __name__ == '__main__':
    shakespeare = 'http://cs.stanford.edu/people/karpathy/' + \
                  'char-rnn/shakespeare_input.txt'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--order', default=6, type=int)
    parser.add_argument('-u', '--url', nargs='+')
    parser.add_argument('-f', '--file', nargs='+')

    args = parser.parse_args()

    from six.moves.urllib import request
    import sys

    def read_urls(*urls):
        text = []
        for url in urls:
            print("Downloading [%s]" % url)
            try:
                req = request.Request(url)
                with request.urlopen(req) as f:
                    for line in f.read().decode('utf-8').split('\r\n'):
                        text += [line + "\n"]
            except ValueError:
                print("Couldn't download [%s]" % url)
        return text

    def read_files(*files):
        import os
        text = []
        for fl in files:
            if os.path.isfile(fl):
                with open(fl, mode='r', encoding='utf-8') as f:
                    print("Reading [%s]" % fl)
                    for line in f.read().split('\n'):
                        text += [line + '\n']
            elif os.path.isdir(fl):
                for ffl in os.listdir(fl):
                    flpath = os.path.join(fl, ffl)
                    if not os.path.isfile(flpath):
                        print("Ignoring [%s]" % flpath)
                        continue
                    with open(flpath, mode='r', encoding='utf-8') as f:
                        print("Reading [%s]" % flpath)
                        for line in f.read().split('\n'):
                            text += [line + '\n']
        return text

    print("Fetching texts")
    text = []
    if args.url:
        text += read_urls(*args.url)
    if args.file:
        text += read_files(*args.file)
    if not text:
        print("No input text, exiting...")
        sys.exit(0)

    model = UnsmoothedLM(order=args.order)

    print("Training on corpus")
    model.train(generate_pairs(text, order=args.order))

    def ensure_res(question, validators, msg, prompt='>>> '):
        print(question)
        res = input(prompt)
        while not any(filter(lambda x: x(res), validators)):
            print(msg)
            res = input(prompt)
        return res

    question = 'generate text (y) or quit (n)?\n'
    msg = 'Sorry, input must be ("y", "n")'
    validators = [lambda x: x in ('y', 'n')]

    res = None
    while res != 'n':
        print("------ Generating text -------")
        print("-----------------------------")
        print(model.generate_text() + "\n")
        print("--- End of Generated text ---")
        print("-----------------------------")
        res = ensure_res(question, validators, msg)

    print("bye!")
    sys.exit(0)
