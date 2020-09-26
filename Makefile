FOLDER=$$(pwd)
IMAGE_NAME=dl-for-har:latest

.PHONY: build
build: # Build docker image
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} .

.PHONY: build-gpu
build-gpu: # Build docker image
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} . -f Dockerfile_gpu

.PHONY: start
start: build # Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --rm -it -v ${FOLDER}:/work -w /work  ${IMAGE_NAME}

.PHONY: start-gpu
start-gpu: build-gpu # Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --gpus all --rm -it -v ${FOLDER}:/work -w /work ${IMAGE_NAME}

.PHONY: start-docker-compose
start-docker-compose: # Start docker container by docker-compose
	echo "Starting container ${IMAGE_NAME}"
	docker-compose up -d --build

.PHONY: jupyter
jupyter:
	poetry run jupyter notebook --port 9000

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-pyc clean-test

mypy:
	poetry run mypy ./*.py models/*.py src/*.py --strict --allow-redefinition --ignore-missing-imports --allow-subclassing-any

black:
	poetry run black ./**/*.py