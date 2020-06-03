import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def dataset_to_TFR(dataset, prefix_target_filename, features_keys, compression=False, save_path='',
                   allow_unbatch=True, num_shards=1):
    """
    :param dataset: dataset object to be saved into tf-records
    :param prefix_target_filename: prefix of target file
    :param features_keys: features keys in order corresponding to dataset output features
    :param compression: if compression required (2x less disk space, ~10x longer saving, similar read time)
    :param save_path: path for tf-records to be saved (must exist)
    :param allow_unbatch: whether to allow unbatch dataset if necessary, usually desired
    :param num_shards: number of separate tf-record files (shards) in which dataset is to be saved
    :return: _
    """
    from tensorflow.python.data.ops.dataset_ops import BatchDataset
    # unbatch to get equal shards num as possible
    dataset = dataset.unbatch() if isinstance(dataset, BatchDataset) and allow_unbatch else dataset
    # compress or not
    options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP) if compression else None

    for num_shard in range(num_shards):
        print(f'Shard {num_shard} processing...')

        partial_dataset = dataset.shard(num_shards=num_shards, index=num_shard)
        shard_filename = os.path.join(save_path, f'{prefix_target_filename}_{num_shard}.tfrecords')

        with tf.io.TFRecordWriter(shard_filename, options=options) as writer:
            for dataset_features in partial_dataset.as_numpy_iterator():
                features_dict = {}
                for dataset_feature, feature_key in zip(dataset_features, features_keys):
                    # 1,2,3D data template
                    if len(dataset_feature.shape) > 0:
                        dataset_feature = dataset_feature.tostring()
                        saving_type_function_wrapper = _bytes_feature
                    # 0D data template
                    else:
                        if str(dataset_feature.dtype).startswith('int'):
                            saving_type_function_wrapper = _int64_feature
                        elif str(dataset_feature.dtype).startswith('float'):
                            saving_type_function_wrapper = _float_feature
                        else:
                            print(f'Unsupported format for {str(dataset_feature.dtype)} data!')
                            continue

                    features_dict[feature_key] = saving_type_function_wrapper(dataset_feature)
                features_tf_object = tf.train.Features(feature=features_dict)
                example_tf_object = tf.train.Example(features=features_tf_object)
                writer.write(example_tf_object.SerializeToString())
            print(f'Shard {num_shard} processing finished.')
            
            
def display_images_and_labels_from_dataset(dataset, examples = 4):    
    fig, axs = plt.subplots(1,examples)
    for idx, (image, label) in enumerate(dataset.take(examples).as_numpy_iterator()):
        axs[idx].imshow(image)
        axs[idx].set_title(f'Class: {label}')
                
def model_parameters_stats(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
def summarize_model(model):
    print(model.summary())
    tf.keras.utils.plot_model(model)
    
def my_saver(_list, _fname):
    with open(_fname, 'wb') as f:
        np.save(f,np.asarray(_list))    
        
def my_reader(_fname):
    with open(_fname, 'rb') as f:
        read_file = np.load(f)
    return read_file