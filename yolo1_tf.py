# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/12$ 17:49$
# @Author  : 陈思成
# @Email   : 821237536@qq.com
# @File    : yolo1_tf$.py
# Description :Yolo V1 by tensorflow。yolo1的预测过程。
# --------------------------------------

import tensorflow as tf
import numpy as np
import cv2

# leaky_relu激活函数
def leaky_relu(x,alpha=0.1):
	return tf.maximum(alpha*x,x)

class Yolo(object):
	##################### 构造函数：初始化yolo中S、B、C参数#################################################################
	def __init__(self,weights_file,input_image,verbose=True):
		# 后面程序打印描述功能的标志位
		self.verbose = verbose

		# 检测超参数
		self.S = 7 # cell数量
		self.B = 2 # 每个网格的边界框数
		self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
						"bus", "car", "cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person", "pottedplant",
						"sheep", "sofa", "train","tvmonitor"]
		self.C = len(self.classes) # 类别数

		# 边界框的中心坐标xy——相对于每个cell左上点的偏移量
		self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B),
												[self.B, self.S, self.S]), [1, 2, 0])
		self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

		self.threshold = 0.2 # 类别置信度分数阈值
		self.iou_threshold = 0.4 # IOU阈值，小于0.4的会过滤掉

		self.max_output_size = 10 # NMS选择的边界框的最大数量

		self.sess = tf.Session()
		self._build_net() # 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
		self._build_detector() # 【2】解析网络的预测结果：先判断预测框类别，再NMS
		self._load_weights(weights_file) # 【3】导入权重文件
		self.detect_from_file(image_file=input_image) # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	####################################################################################################################

	# 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
	def _build_net(self):
		# 打印状态信息
		if self.verbose:
			print("Start to build the network ...")

		# 输入、输出用占位符，因为尺寸一般不会改变
		self.images = tf.placeholder(tf.float32,[None,448,448,3]) # None表示不确定，为了自适应batchsize

		# 搭建网络模型
		net = self._conv_layer(self.images, 1, 64, 7, 2)
		net = self._maxpool_layer(net, 1, 2, 2)
		net = self._conv_layer(net, 2, 192, 3, 1)
		net = self._maxpool_layer(net, 2, 2, 2)
		net = self._conv_layer(net, 3, 128, 1, 1)
		net = self._conv_layer(net, 4, 256, 3, 1)
		net = self._conv_layer(net, 5, 256, 1, 1)
		net = self._conv_layer(net, 6, 512, 3, 1)
		net = self._maxpool_layer(net, 6, 2, 2)
		net = self._conv_layer(net, 7, 256, 1, 1)
		net = self._conv_layer(net, 8, 512, 3, 1)
		net = self._conv_layer(net, 9, 256, 1, 1)
		net = self._conv_layer(net, 10, 512, 3, 1)
		net = self._conv_layer(net, 11, 256, 1, 1)
		net = self._conv_layer(net, 12, 512, 3, 1)
		net = self._conv_layer(net, 13, 256, 1, 1)
		net = self._conv_layer(net, 14, 512, 3, 1)
		net = self._conv_layer(net, 15, 512, 1, 1)
		net = self._conv_layer(net, 16, 1024, 3, 1)
		net = self._maxpool_layer(net, 16, 2, 2)
		net = self._conv_layer(net, 17, 512, 1, 1)
		net = self._conv_layer(net, 18, 1024, 3, 1)
		net = self._conv_layer(net, 19, 512, 1, 1)
		net = self._conv_layer(net, 20, 1024, 3, 1)
		net = self._conv_layer(net, 21, 1024, 3, 1)
		net = self._conv_layer(net, 22, 1024, 3, 2)
		net = self._conv_layer(net, 23, 1024, 3, 1)
		net = self._conv_layer(net, 24, 1024, 3, 1)
		net = self._flatten(net)
		net = self._fc_layer(net, 25, 512, activation=leaky_relu)
		net = self._fc_layer(net, 26, 4096, activation=leaky_relu)
		net = self._fc_layer(net, 27, self.S*self.S*(self.B*5+self.C))

		# 网络输出，[batch,7*7*30]的张量
		self.predicts = net

	# 【2】解析网络的预测结果：先判断预测框类别，再NMS
	def _build_detector(self):
		# 原始图片的宽和高
		self.width = tf.placeholder(tf.float32,name='img_w')
		self.height = tf.placeholder(tf.float32,name='img_h')

		# 网络回归[batch,7*7*30]：
		idx1 = self.S*self.S*self.C
		idx2 = idx1 + self.S*self.S*self.B
		# 1.类别概率[:,:7*7*20]  20维
		class_probs = tf.reshape(self.predicts[0,:idx1],[self.S,self.S,self.C])
		# 2.置信度[:,7*7*20:7*7*(20+2)]  2维
		confs = tf.reshape(self.predicts[0,idx1:idx2],[self.S,self.S,self.B])
		# 3.边界框[:,7*7*(20+2):]  8维 -> (x,y,w,h)
		boxes = tf.reshape(self.predicts[0,idx2:],[self.S,self.S,self.B,4])

		# 将x，y转换为相对于图像左上角的坐标
		# w，h的预测是平方根乘以图像的宽度和高度
		boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S * self.width,
						  (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S * self.height,
						  tf.square(boxes[:, :, :, 2]) * self.width,
						  tf.square(boxes[:, :, :, 3]) * self.height], axis=3)

		# 类别置信度分数：[S,S,B,1]*[S,S,1,C]=[S,S,B,类别置信度C]
		scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)

		scores = tf.reshape(scores, [-1, self.C])  # [S*S*B, C]
		boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]

		# 只选择类别置信度最大的值作为box的类别、分数
		box_classes = tf.argmax(scores, axis=1) # 边界框box的类别
		box_class_scores = tf.reduce_max(scores, axis=1) # 边界框box的分数

		# 利用类别置信度阈值self.threshold，过滤掉类别置信度低的
		filter_mask = box_class_scores >= self.threshold
		scores = tf.boolean_mask(box_class_scores, filter_mask)
		boxes = tf.boolean_mask(boxes, filter_mask)
		box_classes = tf.boolean_mask(box_classes, filter_mask)

		# NMS (不区分不同的类别)
		# 中心坐标+宽高box (x, y, w, h) -> xmin=x-w/2 -> 左上+右下box (xmin, ymin, xmax, ymax)，因为NMS函数是这种计算方式
		_boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
						   boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
		nms_indices = tf.image.non_max_suppression(_boxes, scores,
												   self.max_output_size, self.iou_threshold)
		self.scores = tf.gather(scores, nms_indices)
		self.boxes = tf.gather(boxes, nms_indices)
		self.box_classes = tf.gather(box_classes, nms_indices)

	# 【3】导入权重文件
	def _load_weights(self,weights_file):
		# 打印状态信息
		if self.verbose:
			print("Start to load weights from file:%s" % (weights_file))

		# 导入权重
		saver = tf.train.Saver() # 初始化
		saver.restore(self.sess,weights_file) # saver.restore导入/saver.save保存

	# 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
	# image_file是输入图片文件路径；
	# deteted_boxes_file="boxes.txt"是最后坐标txt；detected_image_file="detected_image.jpg"是检测结果可视化图片
	def detect_from_file(self,image_file,imshow=True,deteted_boxes_file="boxes.txt",
						 detected_image_file="detected_image.jpg"):
		# read image
		image = cv2.imread(image_file)
		img_h, img_w, _ = image.shape
		scores, boxes, box_classes = self._detect_from_image(image)
		predict_boxes = []
		for i in range(len(scores)):
			# 预测框数据为：[概率,x,y,w,h,类别置信度]
			predict_boxes.append((self.classes[box_classes[i]], boxes[i, 0],
								  boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
		self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)


	################# 对应【1】:定义conv/maxpool/flatten/fc层#############################################################
	# 卷积层：x输入；id：层数索引；num_filters：卷积核个数；filter_size：卷积核尺寸；stride：步长
	def _conv_layer(self,x,id,num_filters,filter_size,stride):

		# 通道数
		in_channels = x.get_shape().as_list()[-1]
		# 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
		weight = tf.Variable(tf.truncated_normal([filter_size,filter_size,in_channels,num_filters],mean=0.0,stddev=0.1))
		bias = tf.Variable(tf.zeros([num_filters,])) # 列向量

		# padding, 注意: 不用padding="SAME",否则可能会导致坐标计算错误
		pad_size = filter_size // 2
		pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		x_pad = tf.pad(x, pad_mat)
		conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
		output = leaky_relu(tf.nn.bias_add(conv, bias))

		# 打印该层信息
		if self.verbose:
			print('Layer%d:type=conv,num_filter=%d,filter_size=%d,stride=%d,output_shape=%s'
					%(id,num_filters,filter_size,stride,str(output.get_shape())))

		return output

	# 池化层：x输入；id：层数索引；pool_size：池化尺寸；stride：步长
	def _maxpool_layer(self,x,id,pool_size,stride):
		output = tf.layers.max_pooling2d(inputs=x,
										 pool_size=pool_size,
										 strides=stride,
										 padding='SAME')
		if self.verbose:
			print('Layer%d:type=MaxPool,pool_size=%d,stride=%d,out_shape=%s'
			%(id,pool_size,stride,str(output.get_shape())))
		return output

	# 扁平层：因为接下来会连接全连接层，例如[n_samples, 7, 7, 32] -> [n_samples, 7*7*32]
	def _flatten(self,x):
		tran_x = tf.transpose(x,[0,3,1,2]) # [batch,行,列,通道数channels] -> [batch,通道数channels,列,行]
		nums = np.product(x.get_shape().as_list()[1:]) # 计算的是总共的神经元数量，第一个表示batch数量所以去掉
		return tf.reshape(tran_x,[-1,nums]) # [batch,通道数channels,列,行] -> [batch,通道数channels*列*行],-1代表自适应batch数量

	# 全连接层：x输入；id：层数索引；num_out：输出尺寸；activation：激活函数
	def _fc_layer(self,x,id,num_out,activation=None):
		num_in = x.get_shape().as_list()[-1] # 通道数/维度
		# 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
		weight = tf.Variable(tf.truncated_normal(shape=[num_in,num_out],mean=0.0,stddev=0.1))
		bias = tf.Variable(tf.zeros(shape=[num_out,])) # 列向量
		output = tf.nn.xw_plus_b(x,weight,bias)

		# 正常全连接层是leak_relu激活函数；但是最后一层是liner函数
		if activation:
			output = activation(output)

		# 打印该层信息
		if self.verbose:
			print('Layer%d:type=Fc,num_out=%d,output_shape=%s'
				  % (id, num_out, str(output.get_shape())))
		return output
	####################################################################################################################


	######################## 对应【4】:可视化检测边界框、将obj的分类结果和坐标保存成txt#########################################
	def _detect_from_image(self, image):
		"""Do detection given a cv image"""
		img_h, img_w, _ = image.shape
		img_resized = cv2.resize(image, (448, 448))
		img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
		img_resized_np = np.asarray(img_RGB)
		_images = np.zeros((1, 448, 448, 3), dtype=np.float32)
		_images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
		scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
												   feed_dict={self.images: _images, self.width: img_w,
															  self.height: img_h})
		return scores, boxes, box_classes

	def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
					 detected_image_file=None):
		"""Show the detection boxes"""
		img_cp = image.copy()
		if deteted_boxes_file:
			f = open(deteted_boxes_file, "w")
		#  draw boxes
		for i in range(len(results)):
			x = int(results[i][1])
			y = int(results[i][2])
			w = int(results[i][3]) // 2
			h = int(results[i][4]) // 2
			if self.verbose:
				print("class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f"
					  % (results[i][0],x, y, w, h, results[i][-1]))

				# 中心坐标 + 宽高box(x, y, w, h) -> xmin = x - w / 2 -> 左上 + 右下box(xmin, ymin, xmax, ymax)
				cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

				# 在边界框上显示类别、分数(类别置信度)
				cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1) # puttext函数的背景
				cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

			if deteted_boxes_file:
				# 保存obj检测结果为txt文件
				f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
						str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')
		if imshow:
			cv2.imshow('YOLO_small detection', img_cp)
			cv2.waitKey(1)
		if detected_image_file:
			cv2.imwrite(detected_image_file, img_cp)
		if deteted_boxes_file:
			f.close()
	####################################################################################################################

if __name__ == '__main__':
	yolo_net = Yolo(weights_file='./YOLO_small.ckpt',input_image='./car.PNG')