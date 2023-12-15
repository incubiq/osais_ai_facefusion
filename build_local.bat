

# =========================================
#     build and push docker image
# =========================================

cp ../osais_ai_base/main_common.py ./_temp/
cp ../osais_ai_base/main_fastapi.py ./_temp/
cp ../osais_ai_base/main_flask.py ./_temp/

cp ../osais_ai_base/osais_auth.py ./_temp/
cp ../osais_ai_base/osais_config.py ./_temp/
cp ../osais_ai_base/osais_inference.py ./_temp/
cp ../osais_ai_base/osais_main.py ./_temp/
cp ../osais_ai_base/osais_pricing.py ./_temp/
cp ../osais_ai_base/osais_s3.py ./_temp/
cp ../osais_ai_base/osais_training.py ./_temp/
cp ../osais_ai_base/osais_utils.py ./_temp/

cp ../osais_ai_base/osais.json ./_temp/
cp -r ../osais_ai_base/static ./_temp/
cp -r ../osais_ai_base/templates ./_temp/

cp ./Dockerfile_facefusion ./Dockerfile

docker build -t yeepeekoo/public:ai_facefusion .  
