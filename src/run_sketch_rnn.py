from sketch_rnn.model import SketchRNN
from sketch_rnn.utils import SketchLoader, get_sketch_labels
from sketch_rnn.config import model_file, valid_data_size, test_data_size
import tensorflow as tf


saving_model = True
loading_model = False

def accuracy(prediction, label):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))



if __name__ == '__main__':
    batch_size = 50
    epoch = 5
    dictionary, reverse_dict = get_sketch_labels()
    n_class = len(dictionary)

    sl = SketchLoader(batch_size=50, epoch=5)

    iterator = sl.train_dataset.make_one_shot_iterator()

    sketch, label, sketch_len = iterator.get_next()

    sketchrnn = SketchRNN(sketch, label, sketch_len,
                          n_class=len(dictionary), cell_hidden=[128, ], n_hidden=128)

    batch_acc = accuracy(tf.nn.softmax(sketchrnn.pred), tf.one_hot(label, n_class))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(sketchrnn.cost)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if loading_model:
            saver.restore(sess, model_file)
            print("model restored")
        else:
            tf.global_variables_initializer().run()

        try:
            step = 0
            while True:
                _, l, acc = sess.run([optimizer, sketchrnn.cost, batch_acc])

                if step % 50 == 0:
                    print("Minibatch at step %d ===== loss: %f, acc: %f" % (step, l, acc))

                    valid_iterator = sl.valid_dataset.make_one_shot_iterator()
                    valid_sketch, valid_label, valid_sketch_len = valid_iterator.get_next()
                    valid_pred = sketchrnn.network(valid_sketch, valid_sketch_len, reuse=True)
                    valid_acc = accuracy(tf.nn.softmax(valid_pred), tf.one_hot(valid_label, n_class))

                    try:
                        [acc] = sess.run([valid_acc])
                        print("validation ===== acc: %f" % (acc))
                    except tf.errors.OutOfRangeError:
                        pass

                if (step + 1) % 1000 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1

        except tf.errors.OutOfRangeError:

            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)
