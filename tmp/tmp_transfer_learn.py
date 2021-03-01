import network

batch_shape = (120, 120)
ch_num = 3

model = network.build_unet_model(batch_shape, ch_num, transfer_learn=True)

# for l in model.layers[:38]:
#     l.trainable = False

# model.compile(optimizer='adam',
#                 metrics=['accuracy'],
#                 loss=model.mean_squared_error_difference_learn)

# model.summary()

for l in model.layers:
    print(l.name, l.trainable)