##
##      To build the AI_FACEFUSION docker image
##

# base stuff
FROM yeepeekoo/public:ai_facefusion_

## keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai

## todo : copy all
# COPY ./ai/checkpoints ./ai/checkpoints
COPY ./ai/facefusion ./ai/facefusion
COPY ./ai/runai.py ./ai/runai.py


# push again the base files
COPY ./_temp/static/* ./static
COPY ./_temp/templates/* ./templates
COPY ./_temp/osais.json .
COPY ./_temp/main_fastapi.py .
COPY ./_temp/main_flask.py .
COPY ./_temp/main_common.py .

COPY ./_temp/osais_auth.py .
COPY ./_temp/osais_config.py .
COPY ./_temp/osais_inference.py .
COPY ./_temp/osais_main.py .
COPY ./_temp/osais_pricing.py .
COPY ./_temp/osais_s3.py .
COPY ./_temp/osais_training.py .
COPY ./_temp/osais_utils.py .

# copy OSAIS mapping into AI
COPY ./facefusion.json .
COPY ./_input/warmup.jpg ./_input/warmup.jpg
COPY ./_input/warmup.mp4 ./_input/warmup.mp4

# overload config with those default settings
ENV ENGINE=facefusion

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5311"]
