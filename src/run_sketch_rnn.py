from .sketch_rnn.model import SketchRNN
from .sketch_rnn.utils import SketchLoader, get_sketch_labels
from .sketch_rnn.config import model_file
import tensorflow as tf


saving_model = True
loading_model = False

if __name__ == '__main__':
    batch_size = 50
    epoch = 5
    dictionary, reverse_dict = get_sketch_labels()

    sl = SketchLoader(batch_size=50, epoch=5)

    iterator = sl.train_dataset.make_one_shot_iterator()

    sketch, label, sketch_len = iterator.get_next()

    sketchrnn = SketchRNN(sketch, label, sketch_len, n_class=len(dictionary),
                          cell_hidden=[128,], n_hidden=128)

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
                _, l, acc = sess.run([optimizer, sketchrnn.cost, sketchrnn.accuracy])

                if step % 50 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l))

                if (step + 1) % 1000 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1

        except tf.errors.OutOfRangeError:

            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)
