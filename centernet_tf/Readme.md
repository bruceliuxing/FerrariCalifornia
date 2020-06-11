# centernet_tf(keras 实现)
( 测试 tensorflow-1.14和 tensorflow2.x 均可以)

## 参考：
>- [官方实现](https://github.com/xingyizhou/CenterNet)
>- [xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet)

 本仓库在 [TF2-CenterNet](https://github.com/YukMingLaw/tf2-CenterNet/blob/master/README.md)基础上添加 hourglass 网络定义[keras-centernet](https://github.com/see--/keras-centernet/blob/14bba508c1ff76048945a15dae11218384f2e60c/tests/hourglass_test.py)

## 坑
> tensorflow2.x 中，keras 模型预测时候，用 model.predict(image_data)进行预测时候会出现内存泄露（搭建服务发现一个sess run 之后不会释放，内存一直在涨），排查测试好久才发现是这个 predict函数问题，改成 model(image_data)后无泄露问题。希望官方能注意到，早点修复！
