# CVDL课程项目：基于VQ-VAE和PixelCNN的动漫头像生成模型研究

### 环境依赖：

`pip install -r requirements.txt`

### 使用方法：

`train_vqvae.py`: 训练VQ-VAE模型。如果要更改数据集的路径，在 `utils.py` 里面的 `load_data_and_data_loaders` 函数里面改。

`generate_e_indices.ipynb`: 使用训练好的VQ-VAE模型，生成隐变量序列，存放在 `./data/encoding_indices/` 目录下。

`train_pixelcnn`: 使用刚才生成的隐变量序列训练PixelCNN模型。

`inference.py`: 推理（生成图像）。记得确认里面的模型路径。

（可选） `visualization.ipynb`: 用于可视化VQ-VAE的训练结果。

### 私货

###### 夹带私货

暂时还没想好夹带什么私货
