FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN pip install numpy pandas spicy scikit-learn tqdm rdkit-pypi==2021.9.5.1 nglview -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install torch-sparse==0.6.14 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && pip install torch-geometric==2.0.4

