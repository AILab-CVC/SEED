pip install salesforce-lavis
pip install transformers==4.28.1
pip install pytorch-lightning==1.0.8
pip install --upgrade tensorboard
pip install loralib
pip install kornia==0.6

cd stable_diffusion
pip install -r requirements.txt
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
cd ..
pip install -e stable_diffusion

