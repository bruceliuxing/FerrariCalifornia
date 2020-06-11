python train.py --random-transform --resume checkpoints/tf14/wine/save_model.h5 --gpu 1 --epochs 200 --batch-size 8 --network hourglass --input-size 256 --lr 0.0001 coco data/wine
#python train.py --random-transform --gpu 5 --epochs 200 --batch-size 16 --network hourglass --input-size 384 --lr 0.001 coco data/zhaoshang
#python train.py --random-transform --gpu 3 --epochs 200 --batch-size 16 --network hourglass --input-size 384 --lr 0.001 coco data/nettool
#python train.py --random-transform --multi-scale --resume checkpoints/2020-06-08/save_model.h5 --gpu 0 --epochs 200 --batch-size 8 --network hourglass --input-size 256 --lr 0.0001 coco data/wine/

#pruning for keras model
#python train.py --random-transform --gpu 0 --batch-size 16 --network hourglass --input-size 384  coco data/wine/
