#!/bin/bash

# Make sure you have added the gunicorn dependency
echo "Starting Gunicorn."
exec gunicorn src.app:app --bind 0.0.0.0:8000