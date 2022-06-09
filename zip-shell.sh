#!/bin/bash
if test -s main.zip
then
    rm main.zip
fi

zip -r main.zip app.json flyai_sdk.py gender_class.py main.py model.py prediction.py showm.py requirements.txt
