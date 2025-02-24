Install Steps:

cuda 11.8
conda create -n ProRobo3D python=3.10.16
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install numpy==1.26.4
pip install scipy==1.13.1
pip install scikit-learn==1.4.1.post1
pip install xformers==0.0.25.post1
pip install datasets==2.10.0
pip install matplotlib==3.9.4
pip install open-clip-torch==2.30.0
pip install einops==0.8.0
pip install accelerate==1.2.1
pip install diffusers==0.27.2
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install omegaconf==2.3.0
theia==0.1.0   pip install -e .
spa==0.1   pip install -e .   pip3 install -r requirements.txt
pip install huggingface-hub==0.22.2
pip install opencv-python==4.10.0.82
pip install transformers==4.49.0
 
