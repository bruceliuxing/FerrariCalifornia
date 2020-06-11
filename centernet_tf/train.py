"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date, timedelta
import os
import sys
import tensorflow as tf

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from models.resnet import centernet

import tensorflow_model_optimization as tfmot
import numpy as np

epoch_count = 0

class UpSampleQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
    #  return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]
        return []

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
    #  return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]
        return []

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      #layer.kernel = quantize_weights[0]
        return

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      #layer.activation = quantize_activations[0]
        return

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    model_name = args.coco_path.split('/')[-1]

    #add tensorboard callback
    tensorboard_callback = None
    if args.tensorboard_dir:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.tensorboard_dir, model_name),
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)
    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(os.path.join(args.snapshot_path, model_name))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path, model_name,
                'save_model.h5'
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)
    #open this to adjust lr auto
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='centernet_loss',
        factor=0.1,
        patience=10,
        verbose=1,
        mode='auto',
        min_delta=0.2,
        cooldown=0,
        min_lr=0
    ))
    return callbacks

def create_generators(args):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'input_size': args.input_size,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        misc_effect = MiscEffect(border_value=0)
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    if args.dataset_type == 'coco':
        from generators.coco import CocoGenerator
        model_name = args.coco_path.split('/')[-1]

        train_generator = CocoGenerator(
            args.coco_path,
            '{}_train_all'.format(model_name),
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            multi_scale=args.multi_scale,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator

def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.num_gpus > 1 and parsed_args.batch_size < parsed_args.num_gpus:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.num_gpus > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args

def parse_args(args):
    """
    Parse the arguments.
    """
#    today = str(date.today() + timedelta(days=0))
    today = 'tf14'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    parser.add_argument('--resume', help='Resume training from a snapshot.')

    parser.add_argument('--batch-size', help='Size of the batches.', default=16, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--num_gpus', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.',
                        action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=None)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default='logs/{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')

    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--input-size', help='Rescale the image so the smallest side is min_side.', type=int, default=384)
    parser.add_argument('--multi-scale', help='Multi-Scale training', default=False, action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,default=10)
    parser.add_argument('--network', help='network architecture for model training', default="hourglass")
    parser.add_argument('--lr', help='learning rate for model training', default=0.001)
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))

def train(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create the generators
    train_generator, validation_generator = create_generators(args)

    num_classes = train_generator.num_classes()
    model, prediction_model, debug_model = centernet(num_classes=num_classes, backbone=args.network, input_size=args.input_size,freeze_bn=False)

    # create the model
    print(args.resume)
    if args.resume:
        print('Loading model, this may take a second...')
        model.load_weights(args.resume, by_name=True)

    # compile model
    def get_lr(y_true, y_pred):
        return tf.keras.backend.get_value(model.optimizer.lr)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=float(args.lr)), loss={'centernet_loss': lambda y_true,y_pred:y_pred}, metrics=[get_lr])
    print(model.metrics_names)
    # model.compile(optimizer=SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-5),loss={'centernet_loss': lambda y_true, y_pred:y_pred})

    #model.summary()
    cur_path = os.path.dirname(os.path.abspath(__file__))
    #tf.keras.utils.plot_model(model, to_file=os.path.join("{}/checkpoints/{}_showing.png".format(cur_path, args.network)), show_shapes=True)
    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    model.fit(
        x=train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
#        validation_split=0.1,
#        validation_data=validation_generator,
        validation_freq=1
    )
    print("loading weight for predicrion model...")
    debug_model.load_weights(args.resume, by_name=True, skip_mismatch=True)
    debug_model.save(os.path.join(cur_path, "model_prediction_final.h5"), include_optimizer=False)

    return 0;

def prune(args = None):
    today = str(date.today() + timedelta(days=0))
    import tensorflow_model_optimization as tfmot
    import numpy as np
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = 20

    train_generator, validation_generator = create_generators(args)

    end_step = np.ceil(train_generator.size() / batch_size).astype(np.int32) * epochs
    prune_params = {"pruning_schedule":tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)}
    model_path = "checkpoints/2020-05-26-hourglass/hourglass_384_debugmodel_prediction.h5"
    model = tf.keras.models.load_model(model_path)
    model_for_prune = tfmot.sparsity.keras.prune_low_magnitude(model, **prune_params)
    model_for_prune.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss={'centernet_loss': lambda y_true,y_pred:y_pred})
    model_for_prune.summary()
    prune_log_dir = os.makedirs("log/prune/{}".format(today))
    tf.keras.utils.plot_model(model_for_prune, to_file="{}/{}_showing_prune.png".format(prune_log_dir, args.network), show_shapes=True)
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=prune_log_dir),
    ]
    model_for_prune.fit_generator(generator=train_generator, initial_epoch=0, verbose=1, epochs=epochs, callbacks=callbacks, workers=args.workers, validation_data=validation_generator)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_prune)
    model_for_export.save("{}/{}_{}_prune.h5".format(prune_log_dir, args.network, args.input-size), include_optimizer=False)
    print('Saved pruned Keras model succ')

def quantize(args = None):
    today = str(date.today() + timedelta(days=0))
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = 20

    train_generator, validation_generator = create_generators(args)

    end_step = np.ceil(train_generator.size() / batch_size).astype(np.int32) * epochs
    model_path = "checkpoints/2020-05-26-hourglass/hourglass_384_debugmodel_prediction.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
#    model_for_quantize = tfmot.sparsity.keras.quantize_low_magnitude(model, **quantize_params)

    """
    RuntimeError: Layer kps.0.center.center.center.center.out.upsampleNN:<class 'tensorflow.python.keras.layers.convolutional.UpSampling2D'> is not supported. You can quantize this layer by passing a `tfmot.quantization.keras.QuantizeConfig` instance to the `quantize_annotate_layer` API.

        model_for_quantize = tfmot.quantization.keras.quantize_model(model)
    """
    def apply_quantization_to_upsample(layer):
        if isinstance(layer, tf.keras.layers.UpSampling2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config = UpSampleQuantizeConfig())
        return layer

    annotated_model = tf.keras.models.clone_model(model, clone_function=apply_quantization_to_upsample)
    with tfmot.quantization.keras.quantize_scope({"UpSampleQuantizeConfig":UpSampleQuantizeConfig}):
        model_for_quantize = tfmot.quantization.keras.quantize_apply(annotated_model)

#    quantize_model = tfmot.quantization.keras.quantize_annotate_model(tf.keras.Sequential([
#        tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.UpSampling2D(), UpSampleQuantizeConfig()),
#        tf.keras.layers.Flatten()
#    ]))
#    with tfmot.quantization.keras.quantize_scope({"UpSampleQuantizeConfig":UpSampleQuantizeConfig}):
#        model_for_quantize = tfmot.quantization.keras.quantize_apply(quantize_model)

#    model_for_quantize.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss={'centernet_loss': lambda y_true,y_pred:y_pred})
#    model_for_quantize.summary()
    quantize_log_dir = "log/quantize/{}".format(today)
    if not os.path.exists(quantize_log_dir):
        os.makedirs(quantize_log_dir)
#    tf.keras.utils.plot_model(model_for_quantize, to_file="{}/{}_showing_quantize.png".format(quantize_log_dir, args.network), show_shapes=True)
#    model_for_quantize.fit_generator(generator=train_generator, initial_epoch=0, verbose=1, epochs=epochs, workers=args.workers, validation_data=validation_generator)

    annotated_model.save("{}/{}_{}_quantize.h5".format(quantize_log_dir, args.network, args.input_size), include_optimizer=False)
    print('Saved quantized Keras model succ')


def main():
    train()
#    prune(args="--random-transform --gpu 0 --batch-size 16 --network hourglass --input-size 384  coco data/wine/")
#    prune()
#    quantize()

if __name__ == '__main__':
    main()
