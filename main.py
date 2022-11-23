import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from PIL import Image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, AveragePooling2D, ZeroPadding2D

st.title('CNN Visualizer')
input_image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

selected_layer = st.selectbox('Select a Layer', [Conv2D, MaxPooling2D, AveragePooling2D, Activation, ZeroPadding2D])

apply_layer = None

if selected_layer:
    if selected_layer == Conv2D:
        filter_size = st.number_input('Filters Size (f x f) f in (1-100)', min_value=1, max_value=100, value=3)
        stride = st.number_input('Stride( s x s)  s in (1 - 100)', min_value=1, max_value=100, value=1)
        activation_function = st.selectbox('Activation Function', ['relu', 'sigmoid', 'tanh', 'softmax', 'linear'])
        padding_type = st.selectbox('Padding Type ', ['valid', 'same'])
        around_padding = st.number_input('Around Padding (p x p) p in (1 - 100)', min_value=1, max_value=100, value=1)

        apply_layer = Conv2D(filters=filter_size,
                             kernel_size=filter_size, strides=stride,
                             activation=activation_function,
                             padding=padding_type)

    elif selected_layer == MaxPooling2D or selected_layer == AveragePooling2D:
        pool_size = st.number_input('Pool Size (f x f) f in (1-100)', min_value=1, max_value=100, value=2)
        stride = st.number_input('Stride( s x s)  s in (1 - 100)', min_value=1, max_value=100, value=2)
        padding_type = st.selectbox('Padding Type ', ['valid', 'same'])
        around_padding = st.number_input('Around Padding (p x p) p in (1 - 100)', min_value=0, max_value=100, value=0)

        apply_layer = selected_layer(pool_size=pool_size,
                                     strides=stride,
                                     padding=padding_type)

    elif selected_layer == Activation:
        activation_function = st.selectbox('Activation Function', ['relu', 'sigmoid', 'tanh', 'softmax', 'linear'])

        apply_layer = Activation(activation_function)

    elif selected_layer == ZeroPadding2D:
        around_padding = st.number_input('Around Padding (p x p) p in (1 - 100)', min_value=1, max_value=100, value=1)

        apply_layer = ZeroPadding2D(padding=around_padding)

if input_image is not None and apply_layer is not None:
    st.image(input_image, caption='Uploaded Image.', use_column_width=True, width=50,)
    image = tf.keras.preprocessing.image.load_img(input_image, target_size=(150, 150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    st.write('Input Array Shape: ', input_arr.shape)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    st.write('Input Array Shape: ', input_arr.shape)

    converted_input = apply_layer(input_arr)
    st.write('Output Array Shape: ', converted_input[0].shape)

    st.write('Result after applying the layer')
    output_image = tf.keras.preprocessing.image.array_to_img(converted_input[0])
    st.image(output_image, use_column_width=True, caption='Result after applying the layer', clamp=True, channels='RGB')