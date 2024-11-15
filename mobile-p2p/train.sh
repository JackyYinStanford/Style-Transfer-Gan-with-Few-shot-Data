#!/usr/bin/env sh
#!/bin/bash

cd mobile-p2p

pwd

exec python train.py --direction AtoB $@
