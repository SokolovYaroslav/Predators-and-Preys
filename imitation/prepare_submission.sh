#!/usr/bin/env bash

mv "$1" pred.pth
mv "$2" prey.pth

zip submit.zip submission.py pred.pth prey.pth
cd ../test_submission && unzip -u ../imitation/submit.zip
cd .. && PYTHONPATH=. python test_submission/test_sokolov.py
