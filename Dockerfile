FROM python:3.8.5-buster

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV SHELL /bin/bash -l

ENV TZ Asia/Tokyo

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip

RUN $HOME/.poetry/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH ${PATH}:/root/.poetry/bin:/bin:/usr/local/bin:/usr/bin

CMD ["bash", "-l"]