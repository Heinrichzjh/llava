后台挂起运行
nohup bash run_zero2.sh > training.log 2>&1 &。
查看进度  
tail -f training.log  
  
终止训练的防范  
pkill -f run_zero2.sh  
pkill -f run_show.py  
