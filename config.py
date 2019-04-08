import json
import argparse
import os

class Config():
    def __init__(self, wdir=None, ddir=None, latent_size=None, beta=None):
        self.wdir = wdir
        self.ddir = ddir
        self.latent_size = latent_size
        self.beta = beta

    def save(self, path):
        with open(path, 'w+') as f:
            f.write(json.dumps(self.__dict__))

    def load(self, path):
        with open(path) as f:
            self.__dict__ = json.loads(f.read())

default = Config()
default.load(os.environ['CONFIG'])
