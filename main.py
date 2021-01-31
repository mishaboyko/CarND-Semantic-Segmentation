#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    prob_layer =  graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 =  graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 =  graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 =  graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, prob_layer, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully-connected convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Concept: Section "Special techniques" in the knowledge.md. Full version: scene understanding, FCN-8-Decoder from the classroom
    # We must add 1x1 convolutions on top of the VGG to reduce the number of filters from 4096 to the number of classes for our specific model.
    padding = 'same'
    kernel_size_in = 1
    # kernel_regulizer gives us the Regulizer. If it isn't applied - the weight will become too large and will be prone to overfitting. As a result you'll get [Insufficient Result](./examples/insufficient_result.png)
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size_in, padding = padding,
                                kernel_regularizer = tf.contrib.layers.l2_regulizer(1e-3))

    # perform de-convolution (a.k.a transpose convolution or upsampling)
    kernel_size_out = 4
    stride = 2 # this does the up-sampling by 2
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, kernel_size_out, stride, padding = padding,
                                        kernel_regularizer = tf.contrib.layers.l2_regulizer(1e-3))
    # Question marks indicate, that the dimentions aren't set yet, cause we allow any size of images. 
    # Helps for debugging: what output in the conv_1x1 dimentions are?
    tf.Print(output, [tf.shape(output)[1:3]]) # 1:3 gives you x, y dimentions of the image

    output_upsampled_x2 = tf.layers.conv2d_transpose(output, num_classes, kernel_size_out, stride, padding = padding,
                                        kernel_regularizer = tf.contrib.layers.l2_regulizer(1e-3))
    # ? Which layer? Classroom! Strides 2, 2 & 2, 2 & 8, 8.
    #output_upsampled_x8 = tf.layers.conv2d_transpose(output_upsampled_x2, num_classes, kernel_size_out, stride, padding = padding,
    #                                    kernel_regularizer = tf.contrib.layers.l2_regulizer(1e-3))

    # Do upsampling for other layers. Get insipired by the "Skip connections" section the knowledge.md or the classroom.
    return None
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Concept: knowledge.md & Classroom: FCN-8 - Classification & Loss
    # The output tensor is 4D so we have to reshape it to 2D (flat image, where high = numer of classes and width = amount of pixels).
    # Use adam optimizer, cause it has fewer hyperparams and other things fall in, like decane, learning rate. Also we can speed the back entrophy loss.

    logits = tf.reshape(input, (-1, num_classes))
    return None, None, None
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # Main thing is to understand, what the get_batches_fn() is doing, cause it is doing a lot of heavy-lifting for you.
    for epoch in epochs:
        for image, label in get_batches_fn(batch_size):
            # Perform training
            # Define losses = session.run do this on the trainer optimizer and cross-entrophy loss (same function that we've just implemented)
    pass
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Mischa: Defines, with how many epochs (6?) are we going to work with. Keep batch_size small.
        # Mischa: It's going to create some templates for doing learning rate (float value) and also the correct labels (4D values (batch, higth, width, num_of_classes)).

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_probability, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        NN_final_layer = layers(layer3, layer4, layer7, num_classes)
        # call optimizer

        # TODO: Train NN using the train_nn function
        train_nn()

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
