import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, BatchNormalization

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
def create_tmc_vit_classifier(input_shape, token_emb, patch_size, num_patches,
        projection_dim, transformer_layers, num_heads, transformer_units,
        mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    # Token embedding.
    tokenemb = token_emb(inputs)
    # Create patches.
    patches = Patches(patch_size)(tokenemb)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, 
            dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(18, activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def main():
    # Load data
    X_test = np.load('./data/X_test_200_3.npy')
    X_train = np.load('./data/X_train_200_3.npy')
    y_test = np.load('./data/Y_test_200_3.npy')
    y_train = np.load('./data/Y_train_200_3.npy')

    ## Reshape in order to X_train and test be used as input to the CNNN
    X_train = X_train.reshape(-1, 16, 40, 1)
    X_test = X_test.reshape(-1, 16, 40, 1)

    num_classes = 18
    input_shape = (16, 40, 1)
    image_size1 = 16
    image_size2= 20
    patch_size = 4
    num_patches = (image_size1 // patch_size) * (image_size2 // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    token_emb = keras.Sequential(
        [
            Conv2D(16, (8, 8), activation="relu", padding="same", 
                    input_shape=[X_train.shape[1], X_train.shape[2], 1]),
            BatchNormalization(),
            MaxPooling2D((1, 2)),
            Dropout(0.3),
            Conv2D(32, (4, 4), activation="relu", padding="same"),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (2, 2), activation="relu", padding="same"),
            BatchNormalization(),
        ],
        name="token_emb",
    )

    model = create_tmc_vit_classifier(input_shape, token_emb, patch_size, 
            num_patches, projection_dim, transformer_layers, num_heads, 
            transformer_units, mlp_head_units)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
            metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
            min_delta=0, patience=70, mode='max', restore_best_weights=True)

    print("Starting training...")
    history = model.fit(X_train, y_train, batch_size=128, epochs=500, 
            verbose = 1, validation_data=(X_test, y_test),callbacks=[callback])
    print("Training finished!")

if __name__ == '__main__':
    main()    