#!/usr/bin/env bash
# exit on error

pip install -r requirements.txt
python web/descraibeit/manage.py collectstatic --no-input
python web/descraibeit/manage.py migrate
