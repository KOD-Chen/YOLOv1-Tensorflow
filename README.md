# YOLOv1-Tensorflow<br><br>
## 声明：<br>
更详细的代码解读[Tensorflow实现YOLO1](https://zhuanlan.zhihu.com/p/36819531).<br>
欢迎关注[我的知乎](https://www.zhihu.com/people/chensicheng/posts).<br><br>
## 运行环境：<br>
Python3 + Tensorflow1.5 + OpenCV-python3.3.1 + Numpy1.13<br>
windows和ubuntu环境都可以<br><br>
## 准备工作：<br>
请在[yolo1检测模型](https://pan.baidu.com/s/1mhE0WL6errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)下载训练好的模型YOLO_small.ckpt，并放到同一文件夹下<br><br>
## 文件说明：<br>
1、yolo1_tf.py：程序文件<br>
2、boxes.txt：检测结果的类别和边界框坐标<br><br>
### 运行yolo1_tf.py即可得到效果图：<br>
1、car.PNG：输入的待检测图片<br><br>
![image](https://github.com/Cola-Chen/YOLOv1-Tensorflow/blob/master/car.PNG)<br>
2、detected_image.jpg：检测结果可视化<br><br>
![image](https://github.com/Cola-Chen/YOLOv1-Tensorflow/blob/master/detected_image.jpg)<br>

