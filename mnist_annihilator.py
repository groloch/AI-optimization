import tensorflow as tf


def mnist_annihilator():
    input = tf.keras.layers.Input((28, 28, 1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), dilation_rate=(2, 2), padding='same')(input)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10)(x)

    model = tf.keras.models.Model(input, x)
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    model.fit(x_train, y_train, epochs=60, batch_size=200, validation_split=0.01)


if __name__ == "__main__":
    mnist_annihilator()