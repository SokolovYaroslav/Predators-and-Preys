#!/usr/bin/env bash

mv "$1" pred.pth
mv "$2" prey.pth

zip submit.zip submission.py pred.pth prey.pth
