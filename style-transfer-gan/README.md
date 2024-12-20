## style-transfer Usage Manual



### Align images according to FFHQ image format

```python
# Use FFHQ alignment method to align images in 'raw' folder and save them to 'aligned'
python align_images.py raw/us_comics aligned/us_comics
```



### Project images into latent space

```python
# Reconstruct images from 'aligned' into .npy format latent vectors and save them in 'generated' folder, with 500 reconstruction steps
python project_images.py --num-steps 1000 aligned/us_comics generated/us_comics --network-pkl weights/stylegan2-ffhq-config-f.pkl --vgg16-pkl weights/vgg16_zhang_perceptual.pkl 
```

### Process dataset

The dataset processing method here is the same as StyleGAN2, i.e., first convert the dataset into TFRecords format, then use it to train the model. Note that all images in the dataset should be aligned.

```python
# Convert images in 'untared_raw_image_dir' to .tfrecords and save them in './dataset/' under a folder named 'dataset_name'
python dataset_tool.py create_from_images_raw ./dataset/us_comics aligned/us_comics
```



### Train style-transfer-gan

```python

python run_training.py --num-gpus=2 --data-dir=us_comics --config=config-f --resume-pkl \
weights/stylegan2-ffhq-config-f.pkl --dataset=us_comics_v2 --mirror-augment=true --metric=none \
--total-kimg=12000 --min-h=8 --min-w=8 --res-log2=7 --result-dir=results
```



### Model Fusion and Inference

```python
# fusion and inference are completed together. 'name' and 'prefix' correspond to the saved finetune model from the previous step. 'count' and 'layers' represent the number of finetuned models to be fused and the number of layers to be fused, respectively.
# 'dataset' represents the name of the dataset used for inference visualization
python blend_inference_multiple.py \
--name us_comics_v2 --prefix 00002 --count 000024 000030 000036\
--layers 8 16 32 64 128 256 512 --dataset test_face_female
```



