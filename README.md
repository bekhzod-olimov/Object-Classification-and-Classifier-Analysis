# Object-Classification-and-Classifier-Analysis

Classify objects using DL-based image classification model [rexnet_150](https://github.com/clovaai/rexnet) [(paper)](https://arxiv.org/pdf/2007.00992.pdf), test the model performance on unseen images during training, and perform model analysis using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam).

### Create virtual environment
```python
conda create -n <ENV_NAME> python = 3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Sample data

![Capture](https://user-images.githubusercontent.com/50166164/209258494-82c2972a-babd-429f-904d-272e2255c5f7.PNG)

### Run training with PyTorch
```python
python train.py --batch_size = 64 --lr = 3e-4 --model_name = "efficientnet_b3a"
```

### Run training with PyTorch Lightning (Parallel Training)
```python
python pl_train.py --batch_size = 64 --lr = 3e-4 --model_name = "efficientnet_b3a"
```

### Results

![Capture1](https://user-images.githubusercontent.com/50166164/209258512-b69508a2-0abc-4915-aeef-1fbb1133df63.PNG)

### GradCAM

![Capture](https://user-images.githubusercontent.com/50166164/209279146-a0b81123-7a8f-4cbd-98d6-68f5c17e348b.PNG)
![Capture1](https://user-images.githubusercontent.com/50166164/209279159-2206220e-14bb-4f66-ab62-031e89582ffb.PNG)
![Capture2](https://user-images.githubusercontent.com/50166164/209301961-05786509-bfb2-479d-9a8a-6200e515d28c.PNG)
![Capture3](https://user-images.githubusercontent.com/50166164/209301970-7059fa1e-0259-40a3-a08c-8c468fad95f6.PNG)
![Capture4](https://user-images.githubusercontent.com/50166164/209301976-61cff480-5b13-46f7-9f71-9f04d17719f3.PNG)
![Capture5](https://user-images.githubusercontent.com/50166164/209301983-d51197d7-bd99-4bc0-a2c8-7b303ce4d2bd.PNG)

### Run classfication demo

After completing train process choose a model checkpoint with the best accuracy and do inference using random images from the Internet by running the following script. There are several sample images in sample_ims folder of this repo.

##### Streamlit
```python
streamlit run demo.py --checkpoint_path "path/to/checkpoint"
```

##### Gradio
```python
python gradio_demo.py --checkpoint_path "path/to/checkpoint"
```
