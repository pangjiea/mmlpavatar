1.需要你区分头部和身体的mesh，采样的时候分为face_gs和body_gs，使用@smplx_body_parts_2_faces.json
2.读取数据集smplx信息，包括expression和jawpose参数，获取头部mesh 利用mesh渲染器投影到原图上获取头部掩膜
3.设计网络，头部gs驱动需要expression信号和jawpose信号还有neckpose信号，bodygs则需要目前代码里面有的信号 不需要expression信号和jawpose信号，但也需要neckpose信号
4.保证代码能够正常运行在sq02数据集上面，请你分别完成这两部分，设计新的损失函数，保证解耦(mmlphuman) hello@hello-ThinkStation-P3-Tower:~/code/mmlphuman$ python train.py --config ./config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02
