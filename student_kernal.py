import efficientnet.tfkeras as efn
import tensorflow_addons as tfa

import pandas as pd
import numpy as np
import gc  # garbage collection
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
from tensorflow.python.keras.utils import losses_utils
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import pickle
import random as r
import cv2, os

DEVICE = "TPU"  # or "GPU"
SEED = 42  # USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
FOLDS = 5  # NUMBER OF FOLDS. USE 3, 5, OR 15
IMG_SIZES = [256] * FOLDS
BATCH_SIZES = [32] * FOLDS
EPOCHS = [40] * FOLDS
EFF_NETS = [2] * FOLDS  # STUDENT MODEL
WGTS = [1 / FOLDS] * FOLDS  # WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST
TTA = 81  # TEST TIME AUGMENTATION FACTOR


def seed_all(seed):
    ''' A function to seed everything for getting reproducible results. '''
    r.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    os.environ['TF_KERAS'] = str(seed)
    tf.random.set_seed(seed)


seed_all(SEED)

if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# %matplotlib inline
train = pd.read_csv('/kaggle/input/isic2020-tfrec-256x256-inpainted2/train_final2.csv')
print('Examples WITH Melanoma')
imgs = train.loc[train.target == 1].loc[train.tfrecord == 0].sample(10).image_name.values
plt.figure(figsize=(20, 8))

for i, k in enumerate(imgs):
    img = cv2.imread('/kaggle/input/isic2020-256x256-jpg/stratified_jpg_256/train0/%s.jpg' % k)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

print('Examples WITHOUT Melanoma')
imgs = train.loc[train.target == 0].loc[train.tfrecord == 0].sample(10).image_name.values
plt.figure(figsize=(20, 8))
for i, k in enumerate(imgs):
    img = cv2.imread('/kaggle/input/isic2020-256x256-jpg/stratified_jpg_256/train0/%s.jpg' % k)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

GCS_PATH = [None] * FOLDS
for i, k in enumerate(IMG_SIZES):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('isic2020-tfrec-256x256-inpainted2')

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/results/train*.tfrec')))
files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/results/test*.tfrec')))

ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1, s1, zero,
                                   -s1, c1, zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero,
                                zero, c2, zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
                               zero, one / width_zoom, zero,
                               zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one, zero, height_shift,
                                zero, one, width_shift,
                                zero, zero, one])

    return tf.keras.backend.dot(tf.keras.backend.dot(rotation_matrix, shear_matrix),
                                tf.keras.backend.dot(zoom_matrix, shift_matrix))


def transform(image, DIM=256):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM % 2  # fix for size 331

    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32')
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.keras.backend.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.keras.backend.cast(idx2, dtype='int32')
    idx2 = tf.keras.backend.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def microscopicCutOut(img):
    #     if r.random() <= 0.5:
    Circle = cv2.circle(
        (np.ones(img.shape) * 255).astype(np.uint8),
        (img.shape[0] // 2, img.shape[1] // 2),
        r.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),
        (0, 0, 0),
        -1
    )

    mask = Circle - 255
    img = tf.math.multiply(img, mask)

    return img


CFG = dict(
    sprinkles_mode='normal',
    sprinkles_prob=1,  # probability to spawn a box (between 0-1)
    num_holes=10,  # number of square patches to drop
    side_length=12  # square size
)


# based on: https://www.kaggle.com/benboren/tfrecord-progressive-sprinkles
def make_mask(num_holes, side_length, rows, cols, num_channels):
    '''Builds the mask for all sprinkles.'''
    row_range = tf.tile(tf.range(rows)[..., tf.newaxis], [1, num_holes])
    col_range = tf.tile(tf.range(cols)[..., tf.newaxis], [1, num_holes])
    r_idx = tf.random.uniform([num_holes], minval=0, maxval=rows - 1,
                              dtype=tf.int32)
    c_idx = tf.random.uniform([num_holes], minval=0, maxval=cols - 1,
                              dtype=tf.int32)
    r1 = tf.clip_by_value(r_idx - side_length // 2, 0, rows)
    r2 = tf.clip_by_value(r_idx + side_length // 2, 0, rows)
    c1 = tf.clip_by_value(c_idx - side_length // 2, 0, cols)
    c2 = tf.clip_by_value(c_idx + side_length // 2, 0, cols)
    row_mask = (row_range > r1) & (row_range < r2)
    col_mask = (col_range > c1) & (col_range < c2)

    # Combine masks into one layer and duplicate over channels.
    mask = row_mask[:, tf.newaxis] & col_mask
    mask = tf.reduce_any(mask, axis=-1)
    mask = mask[..., tf.newaxis]
    mask = tf.tile(mask, [1, 1, num_channels])
    return mask


def sprinkles(image, cfg=CFG):
    '''Applies all sprinkles.'''

    num_holes = cfg['num_holes']
    side_length = cfg['side_length']
    mode = cfg['sprinkles_mode']
    PROBABILITY = cfg['sprinkles_prob']

    RandProb = tf.cast(tf.random.uniform([], 0, 1) < PROBABILITY, tf.int32)
    if (RandProb == 0) | (num_holes == 0): return image

    img_shape = tf.shape(image)
    if mode is 'normal':
        rejected = tf.zeros_like(image)
    elif mode is 'salt_pepper':
        num_holes = num_holes // 2
        rejected_high = tf.ones_like(image)
        rejected_low = tf.zeros_like(image)
    elif mode is 'gaussian':
        rejected = tf.random.normal(img_shape, dtype=tf.float32)
    else:
        raise ValueError(f'Unknown mode "{mode}" given.')

    rows = img_shape[0]
    cols = img_shape[1]
    num_channels = img_shape[-1]
    if mode is 'salt_pepper':
        mask1 = make_mask(num_holes, side_length, rows, cols, num_channels)
        mask2 = make_mask(num_holes, side_length, rows, cols, num_channels)
        filtered_image = tf.where(mask1, rejected_high, image)
        filtered_image = tf.where(mask2, rejected_low, filtered_image)
    else:
        mask = make_mask(num_holes, side_length, rows, cols, num_channels)
        filtered_image = tf.where(mask, rejected, image)
    return filtered_image


imgs = train.loc[train.target == 1].loc[train.tfrecord == 0].sample(10).image_name.values
plt.figure(figsize=(20, 8))

for i, k in enumerate(imgs):
    imgpath = ('/kaggle/input/isic2020-256x256-jpg/stratified_jpg_256/train0/%s.jpg' % k)
    with open(imgpath, "rb") as local_file:
        img = local_file.read()
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = transform(img, DIM=256)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.1)
    img = sprinkles(img)
    img = microscopicCutOut(img)
    img = tf.reshape(img, [256, 256, 3])
    plt.imshow(img)
    plt.subplot(2, 5, i + 1);
    plt.axis('off')
    plt.imshow(img)
plt.show()


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(example, tfrec_format)
    label = tf.cast(example['target'], tf.float32)
    return example['image'], label


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64)

    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0


def prepare_image(img, augment=True, dim=256):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    #     img = tf.image.per_image_standardization(img)

    if augment:
        img = transform(img, DIM=dim)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = sprinkles(img)
        img = microscopicCutOut(img)

    img = tf.reshape(img, [dim, dim, 3])
    return img


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files
    n = [int(re.compile(r"-([0-9]*)\_inpaint2.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_dataset(files, augment=False, shuffle=False, repeat=False,
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names),
                    num_parallel_calls=AUTO)

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim),
                                               imgname_or_label),
                num_parallel_calls=AUTO)

    ds = ds.batch(batch_size * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.5,
            temperature=1,
    ):
        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to distillation_loss_fn and 1-alpha to student_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y_ = data
        y = tf.reshape(y_, (-1, 1))
        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            # L_hard
            per_example_loss = self.student_loss_fn(y, student_predictions)
            student_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZES[fold] * REPLICAS)

            teacher_logit = (tf.math.log(teacher_predictions) - tf.math.log(1 - teacher_predictions))
            teacher_predictions_soft = tf.math.sigmoid(teacher_logit / self.temperature)
            student_logit = (tf.math.log(student_predictions) - tf.math.log(1 - student_predictions))
            student_predictions_soft = tf.math.sigmoid(student_logit / self.temperature)

            # L_soft
            per_example_loss_ = self.distillation_loss_fn(
                teacher_predictions_soft,
                student_predictions_soft,
            )
            distillation_loss = tf.nn.compute_average_loss(per_example_loss_,
                                                           global_batch_size=BATCH_SIZES[fold] * REPLICAS)

            #           Lstudent = αKD · Lsoft + (1 − αKD) · Lhard
            loss = (self.alpha) * distillation_loss + (1 - self.alpha) * student_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y_ = data
        y = tf.reshape(y_, (-1, 1))
        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def call(self, data):
        y_pred = self.student(data)
        return y_pred

    @tf.function
    def distributed_train_step(self, dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_test_step(self, dist_inputs):
        return strategy.run(test_step, args=(dist_inputs,))


EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, efn.EfficientNetB4,
        efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]


def get_EFF_NET(dim=128, ef=0, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inp = tf.keras.layers.Input(shape=(dim, dim, 3), name='inp')
    base = EFNS[ef](input_shape=(dim, dim, 3), weights='noisy-student', include_top=False)
    base.trainable = True
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)

    model = tf.keras.Model(inputs=[inp], outputs=[x])
    opti = tfa.optimizers.RectifiedAdam(lr=0.00032, total_steps=10000,
                                        warmup_proportion=0.1, min_lr=1e-7)
    loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.90, gamma=2.0,
                                               reduction=losses_utils.ReductionV2.NONE)

    METRICS = [
        tf.keras.metrics.TruePositives(name='TP'),
        tf.keras.metrics.FalseNegatives(name='FN'),
        tf.keras.metrics.TrueNegatives(name='TN'),
        tf.keras.metrics.FalsePositives(name='FP'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='SEN/Recall'),
        tfa.metrics.F1Score(name='f1score', num_classes=2, average='micro', threshold=0.5),
        tf.keras.metrics.AUC(name='AUC')
    ]
    model.compile(optimizer=opti, loss=loss, metrics=METRICS)
    #     model.summary()

    return model


def get_lr_callback(batch_size=8):
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.3, patience=2, verbose=2,
        mode='max', min_delta=0.001, min_lr=0.000000001
    )

    return lr_callback


VERBOSE = 1
DISPLAY_PLOT = True

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_pred = [];
oof_tar = [];
oof_val = [];
oof_names = [];
oof_folds = []
preds = np.zeros((count_data_items(files_test), 1))
preds1 = np.zeros((count_data_items(files_test), 1))
pred_foldWise = np.asarray([preds] * FOLDS)

for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
    if DEVICE == 'TPU':
        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print('#' * 25);
    print('#### FOLD', fold + 1)
    print('#### Image Size %i and batch_size %i' %
          (IMG_SIZES[fold], BATCH_SIZES[fold] * REPLICAS))

    # CREATE TRAIN AND VALIDATION SUBSETS
    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/_train%.2i*.tfrec' % x for x in idxT])

    np.random.shuffle(files_train);
    print('#' * 25)
    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/_train%.2i*.tfrec' % x for x in idxV])
    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/_test*.tfrec')))

    # SAVE BEST MODEL EACH FOLD
    sv = tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5' % fold, monitor='val_AUC', verbose=0, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch')

    # early stopping with 5 patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', patience=5,
                                                      verbose=2, min_delta=0.0001, restore_best_weights=True)

    a = count_data_items(files_train);
    b = count_data_items(files_valid);
    print('AUGMENTED TRAIN SIZE: ', a, ' * ', TTA, ' = ', a * TTA)
    print('AUGMENTED VALID SIZE: ', b, ' * ', TTA, ' = ', b * TTA)

    # BUILD MODEL
    tf.keras.backend.clear_session()
    with strategy.scope():
        opt = tfa.optimizers.RectifiedAdam(lr=0.00032, total_steps=10000,
                                           warmup_proportion=0.1, min_lr=1e-7)
        opti = tfa.optimizers.Lookahead(opt, sync_period=5, slow_step_size=0.8)
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.90, gamma=2.0,
                                                   reduction=losses_utils.ReductionV2.NONE)

        METRICS = [
            tf.keras.metrics.TruePositives(name='TP'),
            tf.keras.metrics.FalseNegatives(name='FN'),
            tf.keras.metrics.TrueNegatives(name='TN'),
            tf.keras.metrics.FalsePositives(name='FP'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='SEN/Recall'),
            tfa.metrics.F1Score(name='f1score', num_classes=2, average='micro', threshold=0.5),
            tf.keras.metrics.AUC(name='AUC')
        ]
        teacher = get_EFF_NET(dim=IMG_SIZES[fold], ef=5, output_bias=None)
        student = get_EFF_NET(dim=IMG_SIZES[fold], ef=EFF_NETS[fold], output_bias=None)
        # Initialize and compile distiller
        distiller = Distiller(student=student, teacher=teacher)
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)

        teacher.load_weights('../input/b5-9287/b5_9287/fold-2.h5')

        distiller.compile(
            optimizer=opti,
            metrics=METRICS,
            student_loss_fn=loss,
            distillation_loss_fn=tf.keras.losses.KLDivergence(
                reduction=losses_utils.ReductionV2.NONE),
            alpha=0.60,
            temperature=1,
        )

    # Distill teacher to student
    print('Distilling teacher to student....(Training)...')
    history = distiller.fit(
        get_dataset(files_train, augment=True, shuffle=True, repeat=True,
                    dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold]),
        epochs=EPOCHS[fold],
        callbacks=[sv, early_stopping, get_lr_callback(BATCH_SIZES[fold])],
        steps_per_epoch=(count_data_items(files_train)) / BATCH_SIZES[fold] // REPLICAS,
        validation_data=get_dataset(files_valid, augment=False, shuffle=False, repeat=False,
                                    dim=IMG_SIZES[fold]),
        verbose=VERBOSE,
    )
    history.history['val_student_loss'] = np.mean(history.history['val_student_loss'], axis=1)

    distiller.built = True

    print('Loading best student model...')
    distiller.load_weights('fold-%i.h5' % fold)

    # PREDICT OOF USING TTA
    print('Predicting OOF with TTA...')
    ds_valid = get_dataset(files_valid, labeled=False, return_image_names=False, augment=True,
                           repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
    ct_valid = (count_data_items(files_valid));
    STEPS = TTA * ct_valid / BATCH_SIZES[fold] / 4 / REPLICAS
    pred = distiller.predict(ds_valid, steps=STEPS, verbose=VERBOSE)[:TTA * ct_valid, ]

    oof_pred.append(np.mean(pred.reshape((ct_valid, TTA), order='F'), axis=1))

    # GET OOF TARGETS AND NAMES
    ds_valid = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                           labeled=True, return_image_names=True)
    oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))
    oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
    ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                     labeled=False, return_image_names=True)
    oof_names.append(np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

    # REPORT RESULTS
    auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
    oof_val.append(np.max(history.history['val_AUC']))
    print('#### FOLD %i OOF AUC without TTA = %.4f, with TTA = %.4f' % (fold + 1, oof_val[-1], auc))

    # PREDICT TEST with TTA
    print('Predicting Test with TTA...')
    ds_test = get_dataset(files_test, labeled=False, return_image_names=False, augment=True,
                          repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
    ct_test = count_data_items(files_test);
    STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
    pred = distiller.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
    tmp_pred = np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1)
    preds[:, 0] += tmp_pred * WGTS[fold]

    # PREDICT TEST without TTA
    print('Predicting Test without TTA...')
    ds_test = get_dataset(files_test, labeled=False, return_image_names=False, augment=False,
                          repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
    ct_test = count_data_items(files_test);
    STEPS = 1 * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
    pred = distiller.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:1 * ct_test, ]
    tmp_pred1 = np.mean(pred.reshape((ct_test, 1), order='F'), axis=1)
    preds1[:, 0] += tmp_pred1 * WGTS[fold]

    pred_foldWise[fold][:, 0] += tmp_pred

    hist = dict(zip(list(history.history.keys()), np.array(list(history.history.values()))))
    pickle.dump(hist, open("history_fold-%i.p" % (fold + 1), "wb"))
    # PLOT TRAINING
    if DISPLAY_PLOT:
        plt.figure(figsize=(15, 5))
        epoch_new = len(list(history.history.values())[0])
        plt.plot(np.arange(epoch_new), history.history['AUC'], '-o', label='Train AUC', color='#ff7f0e')
        plt.plot(np.arange(epoch_new), history.history['val_AUC'], '-o', label='Val AUC', color='#1f77b4')
        x = np.argmax(history.history['val_AUC']);
        y = np.max(history.history['val_AUC'])
        xdist = plt.xlim()[1] - plt.xlim()[0];
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x, y, s=200, color='#1f77b4');
        plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.4f' % y, size=14)
        plt.ylabel('AUC', size=14);
        plt.xlabel('Epoch', size=14)
        plt.legend(loc=2)
        plt2 = plt.gca().twinx()
        plt2.plot(np.arange(epoch_new), history.history['student_loss'], '-o', label='Train Loss', color='#2ca02c')
        plt2.plot(np.arange(epoch_new), history.history['val_student_loss'], '-o', label='Val Loss', color='#1f77b4')
        x = np.argmin(history.history['val_student_loss']);
        y = np.min(history.history['val_student_loss'])
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x, y, s=200, color='#d62728');
        plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)
        plt.ylabel('Loss', size=14)
        plt.title('FOLD %i - Image Size %i, EFF_NETB%i' % (fold + 1, IMG_SIZES[fold], EFF_NETS[fold]), size=18)
        plt.legend(loc=3)
        plt.show()

    del student;
    del teacher;
    del distiller;
    z = gc.collect()

# COMPUTE OVERALL OOF AUC
oof = np.concatenate(oof_pred);
true = np.concatenate(oof_tar);
names = np.concatenate(oof_names);
folds = np.concatenate(oof_folds)
auc = roc_auc_score(true, oof)
print('Overall OOF AUC with TTA = %.4f' % auc)

# SAVE OOF TO DISK
df_oof = pd.DataFrame(dict(
    image_name=names, target=true, pred=oof, fold=folds))
df_oof.to_csv('oof_Distilled_EFF_NETB2.csv', index=False)
# df_oof.head()

ds = get_dataset(files_test, augment=False, repeat=False, dim=IMG_SIZES[fold],
                 labeled=False, return_image_names=True)

image_names = np.array([img_name.numpy().decode("utf-8")
                        for img, img_name in iter(ds.unbatch())])

submission = pd.DataFrame(dict(image_name=image_names, target=preds1[:, 0]))
submission = submission.sort_values('image_name')
submission.to_csv('without_TTA_submission_augmented_Distilled_EFF_NETB2.csv', index=False)
submission.head()

plt.hist(submission.target, bins=100)
plt.show()

submission = pd.DataFrame(dict(image_name=image_names, target=preds[:, 0]))
submission = submission.sort_values('image_name')
submission.to_csv('with_TTA_submission_augmented_Distilled_EFF_NETB2.csv', index=False)
submission.head()

len(submission)

plt.hist(submission.target, bins=100)
plt.show()

np.save('foldWisePredictions_Distilled_EFF_NETB2.npy', pred_foldWise)
