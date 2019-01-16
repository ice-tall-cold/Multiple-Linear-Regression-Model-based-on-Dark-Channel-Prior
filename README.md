# Train and test on Multiple Linear Regression Model based on DCP
- Run "train_OTS.py" with the OTS dataset to train the Multiple Linear Regression Model and test on the SOTS dataset. You will get the weights and bias under the test PSNR and SSIM values.
- You can access the OTS training dataset and SOTS test dataset through the link [https://sites.google.com/view/reside-dehaze-datasets]
- Run "DCP_recover_RTTS.py" with the RTTS dataset to dehaze all images in RTTS, the output dataset named "result"
- Run "improved_recover_RTTS.py" with the RTTS dataset to dehaze all images in RTTS, the output dataset named "9411result". The weights and bias in this code should be the ones you trained from "train_OTS.py"
- These two images dataset together with original RTTS dataset can be download trhough the link [https://drive.google.com/open?id=1QSpgKxE7Wqu24kMTCClmyCKzg1rhkCBf]

- The original code of DMask R-CNN can be found in [https://github.com/guanlongzhao/dehaze]
- I modified run_DMask.py by adding two paths to access the two new images folder. And I have also modified some parts to make it run properly under my packages version.
