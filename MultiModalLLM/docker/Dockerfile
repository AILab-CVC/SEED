# RUN pip install -U xformers webdataset transformers deepspeed accelerate diffusers["torch"] peft torchvision
# RUN pip uninstall -y apex
# RUN pip install -U torchdata datasets diffusers["torch"] torchinfo
# RUN pip install torchinfo peft==0.2.0
RUN yum install -y java-1.8.0-openjdk
# RUN pip install -U peft diffusers torchinfo 
RUN pip install peft==0.4.0
RUN pip install -U accelerate diffusers torchinfo
# from accelerate.utils import is_npu_available, is_xpu_available 
