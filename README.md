# llama_2_7b_LoRa_QloRa
Project description. 

Setting a system of training local available LLM “meta llama 2/7b”, an implementation was done on client’s own local computer on regular zoom meeting with consultancy and customer service. The flow was to first configure CUDA and CUDNN on his system with NVIDIA RTX 4080 16 GB GPU. The dataset of XMLs was formatted according to the learning method of llama model and then trained by quantizing the model to 4 bit precision to reduce GPU memory consumption, a proper inference code was implemented alongside training code.
