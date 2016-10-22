import argparse
from pathlib import Path
import re
from typing import List

from flask import Flask, render_template, request

from .predict import Model


app = Flask(__name__)
model = None  # type: Model


@app.route('/')
def main():
    phrase = request.args.get('phrase')
    top = []
    if phrase:
        tokens = tokenize(phrase)
        top = model.predict_top(tokens)
    return render_template(
        'main.html',
        phrase=phrase,
        top=top,
        )


def tokenize(phrase: str) -> List[str]:
    # TODO - what was actually used?
    return re.findall('[\w-]+', phrase.lower())


def run():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('vocab')
    arg('--port', type=int, default=8000)
    arg('--host', default='localhost')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    global model
    model = Model(Path(args.model), Path(args.vocab))
    app.run(port=args.port, host=args.host, debug=args.debug)
