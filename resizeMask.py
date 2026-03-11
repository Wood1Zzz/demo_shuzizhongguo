from PIL import Image


imgpath = '/home/danzer/Documents/code/demo_shuzizhongguo/ForgeryAnalysis_Stage_1_Test/Image/234d916686d644e580f3de4752feecf8.jpg'
img = Image.open(imgpath)
print(img.size)  # 输出原始图像尺寸